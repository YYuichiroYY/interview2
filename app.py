import os
import json
import logging
from flask import Flask, request, jsonify
import numpy as np
from janome.tokenizer import Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# .env ファイルから環境変数を読み込む
load_dotenv()

# Flask アプリの生成（静的ファイルは "static" フォルダ内）
app = Flask(__name__, static_folder="static")

# ---------------------------
# 1. JSONファイルから学習用カルテデータを読み込む
def load_json_file():
    try:
        json_file_path = os.path.join(os.path.dirname(__file__), "250226_JSON.txt")
        logging.info(f"JSONファイルの読み込みを試みます: {json_file_path}")
        if not os.path.exists(json_file_path):
            logging.error(f"ファイルが存在しません: {json_file_path}")
            return []
        with open(json_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('\ufeff'):
                content = content[1:]
                logging.info("BOM文字を削除しました")
            sample_data = json.loads(content)
            logging.info(f"JSONデータを正常に読み込みました: {len(sample_data) if isinstance(sample_data, list) else 'オブジェクト'}")
            return sample_data
    except Exception as e:
        logging.error(f"JSONファイルの読み込みエラー: {str(e)}")
        return []

sample_data = load_json_file()
if not sample_data:
    logging.warning("JSONファイルからデータを読み込めませんでした。サンプルデータを使用します。")
    sample_data = [
        {"case_id": "sample1", "chief_complaint": "サンプル主訴1", "raw_text": "サンプルテキスト1"},
        {"case_id": "sample2", "chief_complaint": "サンプル主訴2", "raw_text": "サンプルテキスト2"}
    ]

# ---------------------------
# 2. 日本語用の単語抽出（Janome利用）
tokenizer = Tokenizer()
def extract_tokens(text):
    if not text:
        return set()
    tokens = tokenizer.tokenize(text, wakati=True)
    tokens = [t.lower() for t in tokens]
    return set(tokens)

# ---------------------------
# 3. Sentence Transformer によるエンコーディング準備
texts = [case.get("chief_complaint", "") for case in sample_data]
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)

# ---------------------------
# 4. 類似カルテ検索関数
def find_similar_cases(query, embeddings, sample_data, top_k=3, cutoff=0.65, min_common_words=1):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:min(top_k, len(similarities))]
    query_tokens = extract_tokens(query)
    
    results = []
    for idx in top_indices:
        sim_score = similarities[idx]
        if sim_score < cutoff:
            continue
        case = sample_data[idx]
        candidate_text = case.get("chief_complaint", "") + " " + case.get("raw_text", "")
        candidate_tokens = extract_tokens(candidate_text)
        common_tokens = query_tokens & candidate_tokens
        if len(common_tokens) < min_common_words:
            continue
        results.append({
            "case": case,
            "similarity": sim_score,
            "common_tokens": list(common_tokens)
        })
    return results if results else None

# ---------------------------
# 5. 診断処理エンドポイント (/diagnose)
@app.route("/diagnose", methods=["POST"])
def diagnose():
    logging.info('診断処理関数が呼び出されました')
    try:
        req_body = request.get_json()
        if not req_body:
            return jsonify({"error": "JSON形式のリクエストを送ってください"}), 400
        
        # ここでは、フロントエンドから送られてくる入力を「困りごと」として扱う
        input_query = req_body.get("symptom")
        if not input_query:
            return jsonify({"error": "リクエストに 'symptom' キーが含まれていません"}), 400
        
        # 類似カルテの検索
        similar_cases_info = find_similar_cases(input_query, embeddings, sample_data, top_k=3, cutoff=0.65, min_common_words=1)
        if similar_cases_info is None:
            similar_text = "該当する過去カルテはありません。"
        else:
            similar_text = ""
            for info in similar_cases_info:
                case = info["case"]
                sim_score = info["similarity"]
                common_tokens = info["common_tokens"]
                similar_text += f"【カルテID: {case.get('case_id', 'ID不明')}】\n"
                similar_text += f"主訴: {case.get('chief_complaint', '主訴なし')}\n"
                similar_text += f"カルテ内容: {case.get('raw_text', 'カルテ内容なし')}\n"
                similar_text += f"類似度: {sim_score:.2f}\n"
                similar_text += f"共通単語: {', '.join(common_tokens)}\n\n"
        
        # プロンプトの作成：入力された困りごとをもとに、以下の質問に回答する形で問診項目を作成する
        prompt = f"""あなたは小児科専門医です。{input_query}という困りごとがある患者が来院しました。
以下の質問に回答してください：
1. 今日の食事内容について教えてください。特に、新しく食べたものやアレルギーがある食べ物はありますか？
2. 症状が出現する前に食べたものや摂取したものがあれば教えてください。
3. 過去にアレルギー反応が出た食べ物があれば教えてください。
4. 現在、他に体に異常を感じている症状はありますか？
5. 今回の症状が出現する前に、最後に摂取した食事からの時間を教えてください.
"""
        # OpenAI API キーの取得
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logging.warning("OpenAI APIキーが設定されていません。モックレスポンスを返します。")
            generated_questions = (
                "以下の質問に回答してください：\n"
                "1. 今日の食事内容について教えてください。特に、新しく食べたものやアレルギーがある食べ物はありますか？\n"
                "2. 症状が出現する前に食べたものや摂取したものがあれば教えてください。\n"
                "3. 過去にアレルギー反応が出た食べ物があれば教えてください。\n"
                "4. 現在、他に体に異常を感じている症状はありますか？\n"
                "5. 今回の症状が出現する前に、最後に摂取した食事からの時間を教えてください。"
            )
            final_output = f"主訴: {input_query}\n{generated_questions}\n【類似カルテデータ】\n{similar_text}"
            return jsonify({"result": final_output}), 200

        # ---------------------------
        # 新しい OpenAI SDK のインターフェースを使用して生成AIへリクエスト
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)
        
        # ここで、プロンプトに続けて、上記質問を生成するよう指示
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "あなたは小児科専門医です。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        generated_text = response.choices[0].message.content.strip()
        
        # 最終的な出力は、入力された困りごとと、生成された質問（問診項目）と、類似カルテデータの順に表示
        final_output = f"主訴: {input_query}\n{generated_text}\n【類似カルテデータ】\n{similar_text}"
        return jsonify({"result": final_output}), 200
        
    except Exception as e:
        logging.error(f"エラー発生: {str(e)}")
        return jsonify({"error": f"内部エラー: {str(e)}"}), 500

# ---------------------------
# 6. フロントエンド用のルート（静的ファイルとして index.html を返す）
@app.route("/")
def index():
    return app.send_static_file("index.html")

# ---------------------------
# Flask アプリの起動（Render 用に PORT 環境変数を利用）
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
