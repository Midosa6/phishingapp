# train_model_final.py

import pandas as pd
import tldextract
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------
# 1️⃣ قراءة Dataset
# ---------------------------
df = pd.read_csv(r"C:\Users\Clone\Desktop\New folder\URL dataset.csv")
df.columns = df.columns.str.strip()  # إزالة أي مسافات زائدة

# استخدام الأعمدة الصحيحة: 'url' و 'Type'
df = df[['url', 'type']].copy()
df = df.rename(columns={'type': 'label'})

# تحويل الفئات إلى binary: benign=0, أي نوع آخر=1
df['label'] = df['label'].apply(lambda x: 0 if str(x).lower() in ['benign', 'legitimate'] else 1)

# ---------------------------
# 2️⃣ استخراج ميزات من URL
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
data = pd.concat([features, df['label']], axis=1)

# ---------------------------
# 3️⃣ تقسيم Train / Test
# ---------------------------
X = data.drop('label', axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 4️⃣ تدريب النموذج
# ---------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# 5️⃣ تقييم النموذج
# ---------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---------------------------
# 6️⃣ تجربة URL جديد مع الاحتمال المئوي
# ---------------------------
def check_url_prob(url):
    features_new = pd.DataFrame([extract_features(url)])
    # إذا هناك فئة واحدة فقط بعد التدريب
    if len(model.classes_) == 2:
        prob = model.predict_proba(features_new)[0][1]  # احتمال phishing
    else:
        prob = 1.0 if model.classes_[0] == 1 else 0.0  # 100% phishing أو 0% benign
    percent = round(prob * 100, 2)
    result = "⚠️ Phishing / Malicious" if prob >= 0.5 else "✅ Benign"
    return result, percent

# تجربة تفاعلية
while True:
    url_input = input("Enter a URL to check (or type 'exit' to quit): ")
    if url_input.lower() == "exit":
        break
    result, percent = check_url_prob(url_input)
    print(f"Result: {result} ({percent}% likelihood of phishing)\n")