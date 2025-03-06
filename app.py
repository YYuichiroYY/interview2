import os
import json
import logging
from flask import Flask, request, jsonify
from janome.tokenizer import Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# .env ファイルから環境変数を読み込む
load_dotenv()

# Flask アプリの生成
app = Flask(__name__, static_folder="static")

# ---------------------------
# 1. JSONファイルからカルテデータの読み込み
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
# 2. 形態素解析によるトークン抽出（助詞を除外）
tokenizer = Tokenizer()
def extract_tokens(text):
    tokens = []
    for token in tokenizer.tokenize(text):
        # 助詞を除外
        if "助詞" in token.part_of_speech:
            continue
        base = token.base_form if token.base_form != "*" else token.surface
        tokens.append(base.lower())
    return set(tokens)

# ---------------------------
# 3. シノニム辞書の設定と拡張関数
synonyms = {
    "発熱": {"熱", "微熱", "高熱", "ねつ"},
    "熱": {"発熱", "微熱", "高熱", "ねつ"},
    "咳": {"咳嗽", "せき"},
    "蕁麻疹": {"じんましん", "ぶつぶつ", "皮疹", "かゆみ"},
    "嘔吐": {"嘔", "吐き気", "吐", "吐く"},
    "下痢": {"下痢症", "下痢気味", "水様便"},
    "腹痛": {"腹部痛", "お腹の痛み", "腹の痛み"},
    "鼻汁": {"鼻水", "黄色い鼻", "透明の鼻"},
    "咽頭痛": {"喉痛", "喉の痛み"},
    "経口摂取不良": {"口から摂取できない", "摂食障害", "飲食困難"},
    "持続": {"持続性", "続く", "長引く", "遷延"},
    "腫脹": {"腫れ", "むくみ", "腫み"},
    "意識消失": {"失神", "意識喪失", "気絶"},
    "頭痛": {"頭が痛い", "偏頭痛", "片頭痛", "ヘッドエイク", "頭"}
}

def expand_tokens(tokens):
    expanded = set(tokens)
    for token in tokens:
        if token in synonyms:
            expanded |= synonyms[token]  # 同義語を追加
    return expanded

# ---------------------------
# 4. SentenceTransformerの初期化と個別エンコード
model = SentenceTransformer('all-MiniLM-L6-v2')

# 各カルテの主訴と現病歴のリストを作成
chief_texts = [case.get("chief_complaint", "") for case in sample_data]
raw_texts   = [case.get("raw_text", "") for case in sample_data]

# 個別にエンコード
chief_embeddings = model.encode(chief_texts)
raw_embeddings   = model.encode(raw_texts)

# ---------------------------
# 5. 類似カルテ検索関数
#    主訴80%、現病歴20%の重み付け、総合類似度閾値0.5、主訴部分の共通キーワードが1語以上必要
def find_similar_cases(query, chief_embeddings, raw_embeddings, sample_data, top_k=5, threshold=0.5):
    # クエリのエンコード
    query_embedding = model.encode([query])
    
    # 主訴・現病歴それぞれの類似度計算
    similarities_chief = cosine_similarity(query_embedding, chief_embeddings)[0]
    similarities_raw   = cosine_similarity(query_embedding, raw_embeddings)[0]
    
    results = []
    # クエリのトークン拡張
    query_tokens = expand_tokens(extract_tokens(query))
    
    for idx, case in enumerate(sample_data):
        sim_chief = similarities_chief[idx]
        sim_raw   = similarities_raw[idx]
        weighted_sim = 0.8 * sim_chief + 0.2 * sim_raw
        
        if weighted_sim < threshold:
            continue
        
        # 主訴部分のトークン抽出とシノニム展開
        candidate_tokens = expand_tokens(extract_tokens(case.get("chief_complaint", "")))
        common_tokens = query_tokens & candidate_tokens
        
        if len(common_tokens) < 1:
            continue
        
        results.append({
            "case": case,
            "similarity_chief": sim_chief,
            "similarity_raw": sim_raw,
            "weighted_similarity": weighted_sim,
            "common_tokens": list(common_tokens)
        })
    
    results = sorted(results, key=lambda x: x["weighted_similarity"], reverse=True)
    return results[:top_k] if results else None

# ---------------------------
# 6. 診断処理エンドポイント (/diagnose)
@app.route("/diagnose", methods=["POST"])
def diagnose():
    logging.info('診断処理関数が呼び出されました')
    try:
        req_body = request.get_json()
        if not req_body:
            return jsonify({"error": "JSON形式のリクエストを送ってください"}), 400
        
        input_query = req_body.get("symptom")
        if not input_query:
            return jsonify({"error": "リクエストに 'symptom' キーが含まれていません"}), 400
        
        # 類似カルテの検索（新しい抽出条件を使用）
        similar_cases_info = find_similar_cases(input_query, chief_embeddings, raw_embeddings, sample_data, top_k=5, threshold=0.5)
        
        if similar_cases_info is None:
            similar_text = "該当する過去カルテはありません。"
        else:
            similar_text = ""
            for info in similar_cases_info:
                case = info["case"]
                similar_text += f"【カルテID: {case.get('case_id', 'ID不明')}】\n"
                similar_text += f"主訴: {case.get('chief_complaint', '主訴なし')}\n"
                similar_text += f"現病歴: {case.get('raw_text', '現病歴なし')}\n"
                similar_text += f"重み付き類似度: {info['weighted_similarity']:.2f}\n"
                similar_text += f"共通キーワード（主訴部分）: {', '.join(info['common_tokens'])}\n\n"
        
        # プロンプトの作成
        prompt = f"""あなたは小児科専門医です。{input_query}という主訴の患児が来院しました。
以下は、今回の主訴と類似性が高い主訴で来院した過去の患者のカルテデータです。これらのカルテデータを参考に、以下の指示に従って患者に問診すべき項目を作成してください。
【指示】
目的：今回の症例に必要な情報を、過去のカルテデータに共通して見られる項目を中心に、漏れなく集められるようにすること。
出力形式：実際に患者さんやその保護者に直接尋ねることができるシンプルかつ明確な質問文で提示する。
質問項目の数：最低5項目、必要に応じて最大10項目まで作成する。
【過去カルテデータ】
{similar_text}
"""
        # OpenAI API キーの取得
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logging.warning("OpenAI APIキーが設定されていません。モックレスポンスを返します。")
            mock_questions = [
                "いつから症状が始まりましたか？",
                "症状の強さはどの程度ですか？",
                "以前にも同様の症状はありましたか？",
                "症状が出る状況や時間帯に特徴はありますか？",
                "現在使用している薬は何ですか？"
            ]
            result = {
                "symptom": input_query,
                "similar_cases": similar_text,
                "diagnosis": mock_questions,
                "note": "OpenAI APIキーが設定されていないため、モックデータを返しています"
            }
            return jsonify(result), 200
        
        # OpenAI SDK を用いた問診項目生成リクエスト
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは小児科専門医です。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        generated_text = response.choices[0].message.content.strip()
        
        result = {
            "symptom": input_query,
            "similar_cases": similar_text,
            "diagnosis": generated_text
        }
        return jsonify(result), 200
        
    except Exception as e:
        logging.error(f"エラー発生: {str(e)}")
        return jsonify({"error": f"内部エラー: {str(e)}"}), 500

# ---------------------------
# フロントエンド用のルート（静的ファイルとして index.html を返す）
@app.route("/")
def index():
    return app.send_static_file("index.html")

# ---------------------------
# Flask アプリの起動（PORT環境変数がある場合はその値、なければ5000番ポート）
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
