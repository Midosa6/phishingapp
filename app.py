from flask import Flask, render_template, request # type: ignore
import pandas as pd # type: ignore
import tldextract # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

app = Flask(__name__)

# ---------------------------
# 1️⃣ قراءة CSV الخارجي
# ---------------------------
csv_path = r"C:\Users\Clone\Desktop\New folder\URL dataset.csv"
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()  # إزالة أي مسافات زائدة
df = df[['url', 'type']].copy()
df = df.rename(columns={'type': 'label'})

# تحويل الفئات إلى binary: benign/legitimate=0, أي شيء آخر=1
df['label'] = df['label'].apply(lambda x: 0 if str(x).lower() in ['benign', 'legitimate'] else 1)

# ---------------------------
# 2️⃣ استخراج الميزات
# ---------------------------
def extract_features(url):
    ext = tldextract.extract(url)
    features = {}
    features['url_length'] = len(url)
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_hyphens'] = url.count('-')
    features['num_underscores'] = url.count('_')
    features['num_dots'] = url.count('.')
    features['has_ip'] = int(any(c.isdigit() for c in (ext.subdomain + ext.domain)))
    features['num_subdomains'] = len(ext.subdomain.split('.')) if ext.subdomain else 0
    features['has_at_symbol'] = int('@' in url)
    features['has_https'] = int(url.startswith('https'))
    return features

features = df['url'].apply(lambda x: pd.Series(extract_features(x)))
data_final = pd.concat([features, df['label']], axis=1)

X = data_final.drop('label', axis=1)
y = data_final['label']

# ---------------------------
# 3️⃣ تدريب النموذج
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# دالة التحقق من الرابط
# ---------------------------
def check_url_prob(url):
    features_new = pd.DataFrame([extract_features(url)])
    if len(model.classes_) == 2:
        prob = model.predict_proba(features_new)[0][1]
    else:
        prob = 1.0 if model.classes_[0] == 1 else 0.0
    percent = round(prob * 100, 2)
    result = "⚠️ Phishing / Malicious" if prob >= 0.5 else "✅ Legitimate"
    return result, percent

# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    percent = None
    url_input = None
    if request.method == "POST":
        url_input = request.form["url_input"]
        result, percent = check_url_prob(url_input)
    return render_template("index.html", result=result, percent=percent, url_input=url_input)

if __name__ == "__main__":
    app.run(debug=True)