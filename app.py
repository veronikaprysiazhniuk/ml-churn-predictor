"""
ML Churn Predictor — Customer Retention Intelligence
Predict which customers will leave. Know why. Act before they do.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from io import BytesIO
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ML Churn Predictor",page_icon="◇",layout="wide",initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.stApp{font-family:'Inter',-apple-system,sans-serif;background:#f8fafc}
#MainMenu,header,footer{visibility:hidden}
section[data-testid="stSidebar"]{background:#0c1929;border-right:1px solid #1a2744}
section[data-testid="stSidebar"] .stMarkdown p,section[data-testid="stSidebar"] label{color:#8298b5!important;font-size:13px}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3,section[data-testid="stSidebar"] h4{color:#c8d8ec!important}
.block-container{padding-top:2rem;max-width:1060px}
.rh{padding:0 0 20px;border-bottom:2px solid #1e3a5f;margin-bottom:24px}
.rh-brand{display:flex;align-items:center;gap:8px}
.rh-dot{width:10px;height:10px;background:#7c3aed;border-radius:2px}
.rh-co{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:#64748b}
.rh-title{font-size:24px;font-weight:700;color:#0f172a;margin:10px 0 2px;letter-spacing:-0.5px}
.rh-meta{font-size:13px;color:#64748b}
.es{background:#f0ecff;border:1px solid #c7b9f5;border-radius:10px;padding:18px 22px;margin:16px 0}
.es-label{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#7c3aed;margin-bottom:6px}
.es-text{font-size:14px;color:#1e293b;line-height:1.7}
.es-text b{font-weight:600}
.kr{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:16px 0 24px}
.kc{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px;position:relative;overflow:hidden}
.kc::after{content:'';position:absolute;bottom:0;left:0;right:0;height:3px}
.kc-1::after{background:#7c3aed}.kc-2::after{background:#ef4444}.kc-3::after{background:#10b981}.kc-4::after{background:#3b82f6}
.kc-l{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.7px;color:#64748b;margin-bottom:4px}
.kc-v{font-size:24px;font-weight:700;color:#0f172a;letter-spacing:-0.5px}
.kc-n{font-size:11px;color:#94a3b8;margin-top:3px}
.qs{margin:28px 0 0}
.qb{display:flex;align-items:center;gap:10px;padding-bottom:8px;border-bottom:1px solid #cbd5e1;margin-bottom:14px}
.qn{font-size:12px;font-weight:700;color:#7c3aed}
.qt{font-size:15px;font-weight:600;color:#0f172a}
.co{background:#fff;border:1px solid #e2e8f0;border-left:3px solid #7c3aed;border-radius:0 8px 8px 0;padding:14px 18px;margin:10px 0;font-size:13px;color:#334155;line-height:1.7}
.co.risk{border-left-color:#f59e0b;background:#fffbeb}.co.bad{border-left-color:#ef4444;background:#fef2f2}.co.good{border-left-color:#10b981;background:#f0fdf4}
.co-l{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:3px}
.co .co-l{color:#7c3aed}.co.risk .co-l{color:#d97706}.co.bad .co-l{color:#dc2626}.co.good .co-l{color:#059669}
.ab{background:#0c1929;border-radius:10px;padding:20px 22px;margin:16px 0;color:white}
.ab h4{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#a78bfa;margin-bottom:10px}
.ai{display:flex;gap:10px;padding:7px 0;border-bottom:1px solid #1a2744;font-size:13px;color:#c8d8ec;line-height:1.6}
.ai:last-child{border:none}
.ai-n{color:#a78bfa;font-weight:700;min-width:20px}
.cw{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 12px 4px;margin:8px 0}
.af{margin-top:32px;padding:16px 0;border-top:1px solid #e2e8f0;text-align:center;font-size:11px;color:#94a3b8}
.pt{width:100%;border-collapse:collapse;font-size:13px;margin:8px 0}
.pt th{background:#f1f5f9;padding:10px 12px;text-align:left;font-weight:600;color:#334155;border-bottom:2px solid #e2e8f0;font-size:12px}
.pt td{padding:9px 12px;border-bottom:1px solid #f1f5f9;color:#475569}
.pt tr:hover td{background:#f8fafc}
.pt .num{text-align:right;font-variant-numeric:tabular-nums}
.risk-high{color:#dc2626;font-weight:600}.risk-med{color:#d97706;font-weight:600}.risk-low{color:#059669;font-weight:600}
.metric-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin:12px 0}
.metric-box{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px;text-align:center}
.metric-box .val{font-size:22px;font-weight:700;color:#0f172a}
.metric-box .lbl{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;margin-top:2px}
.stDownloadButton>button{background:#1e3a5f!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important}
@media(max-width:768px){.kr{grid-template-columns:repeat(2,1fr)}}
</style>
""",unsafe_allow_html=True)

COLORS=['#7c3aed','#3b82f6','#10b981','#f59e0b','#ef4444','#06b6d4','#ec4899','#64748b']

def sfig(fig,h=360):
    fig.update_layout(font=dict(family="Inter",color="#334155",size=12),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',margin=dict(l=50,r=24,t=56,b=48),height=h,title_font=dict(size=13,color="#475569"),xaxis=dict(gridcolor='#f1f5f9',linecolor='#e2e8f0',tickfont=dict(size=11,color="#64748b")),yaxis=dict(gridcolor='#f1f5f9',linecolor='#e2e8f0',tickfont=dict(size=11,color="#64748b")),legend=dict(font=dict(size=11)))
    return fig

def generate_demo_data():
    np.random.seed(42)
    n=1000
    customer_id=[f'CUST-{1000+i}' for i in range(n)]
    tenure=np.random.exponential(24,n).clip(1,72).astype(int)
    monthly_spend=np.random.normal(65,25,n).clip(10,200).round(2)
    support_tickets=np.random.poisson(1.5,n)
    login_frequency=np.random.normal(12,5,n).clip(0,30).round(1)
    contract_type=np.random.choice(['Monthly','Annual','Two-Year'],n,p=[0.5,0.35,0.15])
    payment_method=np.random.choice(['Credit Card','Bank Transfer','PayPal','Crypto'],n,p=[0.4,0.3,0.2,0.1])
    plan=np.random.choice(['Basic','Standard','Premium','Enterprise'],n,p=[0.3,0.35,0.25,0.1])
    satisfaction=np.random.normal(3.5,0.8,n).clip(1,5).round(1)
    last_activity_days=np.random.exponential(15,n).clip(0,90).astype(int)
    total_purchases=np.random.poisson(8,n)
    referrals=np.random.poisson(0.5,n)
    # Generate churn with STRONG correlations for good model performance
    churn_score=(
        -1.5
        + 1.8 * (satisfaction < 2.8).astype(float)
        + 0.9 * (satisfaction < 3.5).astype(float)
        + 1.5 * (last_activity_days > 30).astype(float)
        + 0.8 * (last_activity_days > 15).astype(float)
        + 1.2 * (contract_type == 'Monthly').astype(float)
        - 0.8 * (contract_type == 'Two-Year').astype(float)
        - 0.6 * (tenure > 24).astype(float)
        - 0.4 * (tenure > 12).astype(float)
        + 0.9 * (support_tickets > 3).astype(float)
        + 0.5 * (support_tickets > 1).astype(float)
        - 0.7 * (plan == 'Enterprise').astype(float)
        - 0.4 * (plan == 'Premium').astype(float)
        + 0.6 * (monthly_spend < 35).astype(float)
        - 0.5 * (monthly_spend > 80).astype(float)
        + 0.7 * (login_frequency < 8).astype(float)
        - 0.3 * (referrals > 1).astype(float)
        + np.random.normal(0, 0.5, n)
    )
    churn_prob = 1 / (1 + np.exp(-churn_score))
    churned=(np.random.random(n)<churn_prob).astype(int)
    return pd.DataFrame({
        'Customer_ID':customer_id,'Tenure_Months':tenure,'Monthly_Spend':monthly_spend,
        'Support_Tickets':support_tickets,'Login_Frequency':login_frequency,
        'Contract_Type':contract_type,'Payment_Method':payment_method,'Plan':plan,
        'Satisfaction_Score':satisfaction,'Days_Since_Last_Activity':last_activity_days,
        'Total_Purchases':total_purchases,'Referrals_Made':referrals,'Churned':churned
    })

def make_excel(df,results,risk_df,recs):
    out=BytesIO()
    with pd.ExcelWriter(out,engine='openpyxl') as w:
        df.to_excel(w,sheet_name='Customer Data',index=False)
        if risk_df is not None:risk_df.to_excel(w,sheet_name='Risk Scores',index=False)
        pd.DataFrame([results]).to_excel(w,sheet_name='Model Performance',index=False)
        pd.DataFrame([{'#':i+1,'Action':r} for i,r in enumerate(recs)]).to_excel(w,sheet_name='Recommendations',index=False)
    return out.getvalue()

# Sidebar
with st.sidebar:
    st.markdown('<div style="padding:0.5rem 0 1rem"><div style="display:flex;align-items:center;gap:8px"><div style="width:8px;height:8px;background:#7c3aed;border-radius:2px"></div><span style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;color:#a78bfa">ML Churn Predictor</span></div><p style="font-size:16px;font-weight:600;color:#e2e8f0;margin:8px 0 0">Retention Intelligence</p></div>',unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### Data source")
    uploaded_file=st.file_uploader("",type=['csv','xlsx'],label_visibility="collapsed")
    use_demo=st.checkbox("Load sample dataset",help="SaaS customer churn data (1,000 customers)")
    st.markdown("---")
    st.markdown("#### Model settings")
    test_size=st.slider("Test set size",10,40,20,5,help="% of data for testing")
    model_type=st.selectbox("Algorithm",["Random Forest","Logistic Regression","Both (compare)"])
    st.markdown("---")
    st.markdown('<p style="font-size:10px;color:#475569;line-height:1.5">ML models trained in-browser.<br>No data stored or shared.</p>',unsafe_allow_html=True)

# Data
df=None
if use_demo:df=generate_demo_data()
elif uploaded_file:
    try:df=pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    except Exception as e:st.error(f"Error: {e}")

if df is None:
    st.markdown('<div class="rh"><div class="rh-brand"><div class="rh-dot"></div><span class="rh-co">ML Churn Predictor</span></div><div class="rh-title">Customer Retention Intelligence</div><div class="rh-meta">Predict which customers will leave. Know why. Act before they do.</div></div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1:st.markdown('<div style="padding:20px 0"><p style="font-size:14px;font-weight:600;color:#0f172a;margin-bottom:6px">1. Upload customer data</p><p style="font-size:13px;color:#64748b;line-height:1.6">CSV or Excel with customer attributes and a churn/status column. Works with any dataset structure.</p></div>',unsafe_allow_html=True)
    with c2:st.markdown('<div style="padding:20px 0"><p style="font-size:14px;font-weight:600;color:#0f172a;margin-bottom:6px">2. Train ML models</p><p style="font-size:13px;color:#64748b;line-height:1.6">Random Forest and Logistic Regression trained automatically. Compare accuracy, precision, recall side by side.</p></div>',unsafe_allow_html=True)
    with c3:st.markdown('<div style="padding:20px 0"><p style="font-size:14px;font-weight:600;color:#0f172a;margin-bottom:6px">3. Get risk scores & act</p><p style="font-size:13px;color:#64748b;line-height:1.6">Every customer gets a churn probability. See which factors drive churn. Get revenue-impact analysis and retention recommendations.</p></div>',unsafe_allow_html=True)
    st.info("Use the sidebar to upload customer data or check **Load sample dataset** to explore.")
else:
    # Find target column
    target_col=None
    for c in df.columns:
        if c.lower() in ['churned','churn','is_churned','status','target','label','left','attrition']:
            target_col=c;break
    if target_col is None:
        binary_cols=[c for c in df.columns if df[c].nunique()==2 and pd.api.types.is_numeric_dtype(df[c])]
        if binary_cols:target_col=binary_cols[0]
    if target_col is None:
        st.error("Could not find a target column (churn/status). Please ensure your data has a binary column indicating churn.");st.stop()

    # Prepare features
    id_cols=[c for c in df.columns if 'id' in c.lower() or 'name' in c.lower() or c==target_col]
    feature_cols=[c for c in df.columns if c not in id_cols]
    df_model=df[feature_cols].copy()

    # Encode categoricals
    le_dict={}
    for c in df_model.select_dtypes(include=['object','category']).columns:
        le=LabelEncoder();df_model[c]=le.fit_transform(df_model[c].astype(str));le_dict[c]=le

    # Handle missing
    df_model=df_model.fillna(df_model.median(numeric_only=True))

    X=df_model.drop(columns=[target_col] if target_col in df_model.columns else [])
    y=df[target_col]

    # Split (unscaled for tree models, scaled for linear)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size/100,random_state=42,stratify=y)

    scaler=StandardScaler()
    X_train_scaled=pd.DataFrame(scaler.fit_transform(X_train),columns=X.columns,index=X_train.index)
    X_test_scaled=pd.DataFrame(scaler.transform(X_test),columns=X.columns,index=X_test.index)
    X_all_scaled=pd.DataFrame(scaler.transform(X),columns=X.columns,index=X.index)

    # Train models
    models={}
    results={}
    if model_type in["Random Forest","Both (compare)"]:
        rf=RandomForestClassifier(n_estimators=200,random_state=42,max_depth=10,min_samples_leaf=5,class_weight='balanced')
        rf.fit(X_train,y_train)
        rf_pred=rf.predict(X_test);rf_prob=rf.predict_proba(X_test)[:,1]
        models['Random Forest']={'model':rf,'pred':rf_pred,'prob':rf_prob,'scaled':False}
        results['Random Forest']={'Accuracy':accuracy_score(y_test,rf_pred),'Precision':precision_score(y_test,rf_pred,zero_division=0),'Recall':recall_score(y_test,rf_pred,zero_division=0),'F1':f1_score(y_test,rf_pred,zero_division=0),'AUC':roc_auc_score(y_test,rf_prob)}
    if model_type in["Logistic Regression","Both (compare)"]:
        lr=LogisticRegression(random_state=42,max_iter=1000,class_weight='balanced')
        lr.fit(X_train_scaled,y_train)
        lr_pred=lr.predict(X_test_scaled);lr_prob=lr.predict_proba(X_test_scaled)[:,1]
        models['Logistic Regression']={'model':lr,'pred':lr_pred,'prob':lr_prob,'scaled':True}
        results['Logistic Regression']={'Accuracy':accuracy_score(y_test,lr_pred),'Precision':precision_score(y_test,lr_pred,zero_division=0),'Recall':recall_score(y_test,lr_pred,zero_division=0),'F1':f1_score(y_test,lr_pred,zero_division=0),'AUC':roc_auc_score(y_test,lr_prob)}

    best_name=max(results,key=lambda k:results[k]['AUC'])
    best=models[best_name]
    best_results=results[best_name]

    # Full predictions
    X_for_pred=X_all_scaled if best.get('scaled') else X
    all_prob=best['model'].predict_proba(X_for_pred)[:,1]
    df['Churn_Probability']=all_prob.round(3)
    df['Risk_Level']=pd.cut(all_prob,bins=[0,0.3,0.6,1.0],labels=['Low','Medium','High'])

    churn_rate=y.mean()*100
    total_customers=len(df)
    high_risk=len(df[df['Risk_Level']=='High'])
    avg_spend=df['Monthly_Spend'].mean() if 'Monthly_Spend' in df.columns else 0
    revenue_at_risk=high_risk*avg_spend*12 if avg_spend>0 else high_risk

    recs=[]

    # ─── HEADER ───
    st.markdown(f'<div class="rh"><div class="rh-brand"><div class="rh-dot"></div><span class="rh-co">ML Churn Predictor</span></div><div class="rh-title">Customer Retention Report</div><div class="rh-meta">{datetime.now().strftime("%B %d, %Y")} · {total_customers:,} customers · {best_name} model · {best_results["AUC"]:.1%} AUC</div></div>',unsafe_allow_html=True)

    # ─── EXEC SUMMARY ───
    rev_text=f"Estimated annual revenue at risk: <b>${revenue_at_risk:,.0f}</b>." if avg_spend>0 else ""
    st.markdown(f'<div class="es"><div class="es-label">Executive summary</div><div class="es-text">Current churn rate is <b>{churn_rate:.1f}%</b> across {total_customers:,} customers. The ML model identifies <b>{high_risk} high-risk customers</b> ({high_risk/total_customers*100:.0f}%) likely to leave within the next period. {rev_text} Model accuracy: <b>{best_results["AUC"]:.1%} AUC</b>.</div></div>',unsafe_allow_html=True)

    # ─── KPIs ───
    med_risk=len(df[df['Risk_Level']=='Medium'])
    low_risk=len(df[df['Risk_Level']=='Low'])
    st.markdown(f'<div class="kr"><div class="kc kc-1"><div class="kc-l">Churn rate</div><div class="kc-v">{churn_rate:.1f}%</div><div class="kc-n">{int(y.sum()):,} of {total_customers:,}</div></div><div class="kc kc-2"><div class="kc-l">High risk</div><div class="kc-v">{high_risk}</div><div class="kc-n">{high_risk/total_customers*100:.0f}% of customers</div></div><div class="kc kc-3"><div class="kc-l">Model AUC</div><div class="kc-v">{best_results["AUC"]:.1%}</div><div class="kc-n">{best_name}</div></div><div class="kc kc-4"><div class="kc-l">Revenue at risk</div><div class="kc-v">{"${:,.0f}".format(revenue_at_risk) if avg_spend>0 else "N/A"}</div><div class="kc-n">Annual estimate</div></div></div>',unsafe_allow_html=True)

    # ═══ Q1: MODEL PERFORMANCE ═══
    st.markdown('<div class="qs"><div class="qb"><span class="qn">Q1</span><span class="qt">How reliable is the prediction?</span></div></div>',unsafe_allow_html=True)

    if len(results)>1:
        # Comparison table
        rows=""
        for name,r in results.items():
            is_best=" ★" if name==best_name else ""
            rows+=f'<tr><td><b>{name}{is_best}</b></td><td class="num">{r["Accuracy"]:.1%}</td><td class="num">{r["Precision"]:.1%}</td><td class="num">{r["Recall"]:.1%}</td><td class="num">{r["F1"]:.1%}</td><td class="num"><b>{r["AUC"]:.1%}</b></td></tr>'
        st.markdown(f'<div class="cw" style="padding:16px"><p style="font-size:12px;font-weight:600;color:#7c3aed;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px">Model comparison</p><table class="pt"><tr><th>Model</th><th style="text-align:right">Accuracy</th><th style="text-align:right">Precision</th><th style="text-align:right">Recall</th><th style="text-align:right">F1</th><th style="text-align:right">AUC</th></tr>{rows}</table></div>',unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="metric-grid"><div class="metric-box"><div class="val">{best_results["Accuracy"]:.1%}</div><div class="lbl">Accuracy</div></div><div class="metric-box"><div class="val">{best_results["Precision"]:.1%}</div><div class="lbl">Precision</div></div><div class="metric-box"><div class="val">{best_results["Recall"]:.1%}</div><div class="lbl">Recall</div></div><div class="metric-box"><div class="val">{best_results["AUC"]:.1%}</div><div class="lbl">AUC Score</div></div></div>',unsafe_allow_html=True)

    auc=best_results['AUC']
    q_class='good' if auc>0.8 else 'co' if auc>0.7 else 'risk'
    q_text='Strong predictive power — model reliably distinguishes churners from retained customers.' if auc>0.8 else 'Moderate predictive power — useful for risk segmentation but consider adding more features.' if auc>0.7 else 'Limited predictive power — additional customer data may be needed for reliable predictions.'
    st.markdown(f'<div class="co {q_class}"><div class="co-l">Model assessment</div>{best_name} achieves <b>{auc:.1%} AUC</b>. {q_text}</div>',unsafe_allow_html=True)

    # Confusion Matrix + ROC Curve side by side
    cm_col1,cm_col2=st.columns(2)

    # Confusion Matrix
    cm=confusion_matrix(y_test,best['pred'])
    labels=['Retained','Churned']
    cm_text=[[str(v) for v in row] for row in cm]
    fig_cm=go.Figure(go.Heatmap(z=cm[::-1],x=labels,y=labels[::-1],text=cm_text[::-1],texttemplate='%{text}',textfont=dict(size=18,color='white'),colorscale=[[0,'#c4b5fd'],[1,'#7c3aed']],showscale=False))
    fig_cm.update_layout(title='Confusion matrix',xaxis_title='Predicted',yaxis_title='Actual',margin=dict(l=50,r=24,t=56,b=56),height=340,font=dict(family="Inter",color="#334155",size=12),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    with cm_col1:
        st.markdown('<div class="cw">',unsafe_allow_html=True);st.plotly_chart(fig_cm,use_container_width=True);st.markdown('</div>',unsafe_allow_html=True)

    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr,tpr,_=roc_curve(y_test,best['prob'])
    fig_roc=go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',line=dict(color='#7c3aed',width=2.5),name=f'{best_name} (AUC={auc:.3f})',fill='tozeroy',fillcolor='rgba(124,58,237,0.08)'))
    fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',line=dict(color='#94a3b8',width=1,dash='dot'),name='Random (AUC=0.500)',showlegend=True))
    fig_roc.update_layout(title='ROC curve',xaxis_title='False positive rate',yaxis_title='True positive rate',legend=dict(orientation='h',yanchor='bottom',y=1.02,font=dict(size=10)),margin=dict(l=50,r=24,t=56,b=48),height=340,font=dict(family="Inter",color="#334155",size=12),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',xaxis=dict(gridcolor='#f1f5f9',linecolor='#e2e8f0',tickfont=dict(size=11,color="#64748b")),yaxis=dict(gridcolor='#f1f5f9',linecolor='#e2e8f0',tickfont=dict(size=11,color="#64748b")))
    with cm_col2:
        st.markdown('<div class="cw">',unsafe_allow_html=True);st.plotly_chart(fig_roc,use_container_width=True);st.markdown('</div>',unsafe_allow_html=True)

    # Confusion matrix conclusion
    tn,fp,fn,tp=cm.ravel()
    st.markdown(f'<div class="co"><div class="co-l">Prediction breakdown</div>The model correctly identifies <b>{tp}</b> churners (true positives) and <b>{tn}</b> retained customers (true negatives). It misses <b>{fn}</b> churners (false negatives) and incorrectly flags <b>{fp}</b> retained customers as at-risk (false positives). {"The low false-negative rate means few churners slip through undetected." if fn<tp else "Consider tuning the threshold to catch more churners — currently some are being missed."}</div>',unsafe_allow_html=True)

    # ═══ Q2: WHY DO CUSTOMERS CHURN? ═══
    st.markdown('<div class="qs"><div class="qb"><span class="qn">Q2</span><span class="qt">What drives customer churn?</span></div></div>',unsafe_allow_html=True)

    if hasattr(best['model'],'feature_importances_'):
        imp=pd.Series(best['model'].feature_importances_,index=X.columns).sort_values(ascending=True).tail(10)
    else:
        imp=pd.Series(np.abs(best['model'].coef_[0]),index=X.columns).sort_values(ascending=True).tail(10)

    fig_imp=go.Figure(go.Bar(y=imp.index,x=imp.values,orientation='h',marker_color=['#7c3aed' if i>=len(imp)-3 else '#c4b5fd' for i in range(len(imp))],text=[f'{v:.1%}' for v in imp.values/imp.values.sum()],textposition='auto',textfont=dict(size=11,color='white')))
    fig_imp.update_layout(title='Feature importance — what predicts churn',yaxis_title='',xaxis=dict(showgrid=False,showticklabels=False))
    st.markdown('<div class="cw">',unsafe_allow_html=True);st.plotly_chart(sfig(fig_imp),use_container_width=True);st.markdown('</div>',unsafe_allow_html=True)

    top3=imp.tail(3).index.tolist()[::-1]
    st.markdown(f'<div class="co"><div class="co-l">Key churn drivers</div>The top 3 factors predicting churn are: <b>{top3[0].replace("_"," ")}</b>, <b>{top3[1].replace("_"," ")}</b>, and <b>{top3[2].replace("_"," ")}</b>. Focus retention efforts on improving these metrics for at-risk customers.</div>',unsafe_allow_html=True)
    recs.append(f"Address top churn driver: {top3[0].replace('_',' ')}. Customers with poor scores on this metric are most likely to leave.")

    # Partial dependence plots for top 2 features
    st.markdown('<p style="font-size:13px;font-weight:600;color:#334155;margin:16px 0 8px">How do top features affect churn probability?</p>',unsafe_allow_html=True)
    pd_col1,pd_col2=st.columns(2)
    for idx,feat in enumerate(top3[:2]):
        if pd.api.types.is_numeric_dtype(X[feat]):
            # Create bins for the feature using original (unscaled) data
            feat_orig=df[feat] if feat in df.columns else X[feat]
            bins=np.linspace(feat_orig.min(),feat_orig.max(),20)
            bin_probs=[]
            for b in bins:
                mask=feat_orig<=b
                if mask.sum()>10:
                    bin_probs.append(df.loc[mask,'Churn_Probability'].mean())
                else:
                    bin_probs.append(np.nan)
            fig_pd=go.Figure()
            fig_pd.add_trace(go.Scatter(x=bins,y=bin_probs,mode='lines+markers',line=dict(color='#7c3aed',width=2.5),marker=dict(size=4),fill='tozeroy',fillcolor='rgba(124,58,237,0.06)',name='Churn probability'))
            fig_pd.add_hline(y=churn_rate/100,line_dash="dot",line_color="#94a3b8",annotation_text=f"Avg: {churn_rate:.0f}%")
            fig_pd.update_layout(title=f'Churn probability vs {feat.replace("_"," ")}',xaxis_title=feat.replace("_"," "),yaxis_title='Churn probability',yaxis=dict(tickformat='.0%'))
            col=pd_col1 if idx==0 else pd_col2
            with col:
                st.markdown('<div class="cw">',unsafe_allow_html=True);st.plotly_chart(sfig(fig_pd,300),use_container_width=True);st.markdown('</div>',unsafe_allow_html=True)

    # Partial dependence conclusion
    st.markdown(f'<div class="co"><div class="co-l">Impact analysis</div>The charts above show how churn probability changes as each factor varies. Use these to set <b>intervention thresholds</b> — for example, if churn spikes when {top3[0].replace("_"," ")} crosses a certain value, trigger a retention action at that point.</div>',unsafe_allow_html=True)

    # ═══ Q3: RISK SEGMENTATION ═══
    st.markdown('<div class="qs"><div class="qb"><span class="qn">Q3</span><span class="qt">Which customers need attention now?</span></div></div>',unsafe_allow_html=True)

    # Risk distribution chart
    risk_counts=df['Risk_Level'].value_counts()
    fig_risk=go.Figure(go.Pie(labels=['High Risk','Medium Risk','Low Risk'],values=[risk_counts.get('High',0),risk_counts.get('Medium',0),risk_counts.get('Low',0)],hole=0.6,marker=dict(colors=['#ef4444','#f59e0b','#10b981']),textinfo='label+value+percent',textposition='outside',textfont=dict(size=12)))
    fig_risk.update_layout(title='Customer risk segmentation',showlegend=False,annotations=[dict(text=f'{high_risk}<br>High Risk',x=0.5,y=0.5,font_size=16,font_color='#ef4444',showarrow=False)])
    ch1,ch2=st.columns(2)
    with ch1:
        st.markdown('<div class="cw">',unsafe_allow_html=True);st.plotly_chart(sfig(fig_risk,340),use_container_width=True);st.markdown('</div>',unsafe_allow_html=True)

    # Churn probability distribution
    with ch2:
        fig_dist=go.Figure()
        fig_dist.add_trace(go.Histogram(x=df[df[target_col]==0]['Churn_Probability'],nbinsx=30,name='Retained',marker_color='#10b981',opacity=0.7))
        fig_dist.add_trace(go.Histogram(x=df[df[target_col]==1]['Churn_Probability'],nbinsx=30,name='Churned',marker_color='#ef4444',opacity=0.7))
        fig_dist.update_layout(title='Churn probability distribution',barmode='overlay',xaxis_title='Churn probability',yaxis_title='Customers',legend=dict(orientation='h',yanchor='bottom',y=1.02))
        st.markdown('<div class="cw">',unsafe_allow_html=True);st.plotly_chart(sfig(fig_dist,340),use_container_width=True);st.markdown('</div>',unsafe_allow_html=True)

    # High risk customers table
    high_risk_df=df[df['Risk_Level']=='High'].sort_values('Churn_Probability',ascending=False).head(10)
    display_cols=[c for c in['Customer_ID','Churn_Probability','Monthly_Spend','Tenure_Months','Satisfaction_Score','Plan','Contract_Type'] if c in high_risk_df.columns][:7]
    if len(display_cols)>0:
        rows=""
        for _,r in high_risk_df[display_cols].iterrows():
            cells="".join([f'<td class="num">{v}</td>' if isinstance(v,(int,float)) else f'<td>{v}</td>' for v in r.values])
            rows+=f'<tr>{cells}</tr>'
        headers="".join([f'<th style="text-align:{"right" if pd.api.types.is_numeric_dtype(high_risk_df[c]) else "left"}">{c.replace("_"," ")}</th>' for c in display_cols])
        st.markdown(f'<div class="cw" style="padding:16px"><p style="font-size:12px;font-weight:600;color:#dc2626;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px">Top 10 highest-risk customers</p><table class="pt"><tr>{headers}</tr>{rows}</table></div>',unsafe_allow_html=True)

    if avg_spend>0:
        st.markdown(f'<div class="co bad"><div class="co-l">Revenue impact</div><b>{high_risk} high-risk customers</b> represent <b>${revenue_at_risk:,.0f}</b> in annual revenue. If even 50% of these churn, the business loses <b>${revenue_at_risk*0.5:,.0f}/year</b>. Targeted retention campaigns on this group could save a significant portion.</div>',unsafe_allow_html=True)
        recs.append(f"Launch targeted retention campaign for {high_risk} high-risk customers. Potential annual revenue saved: ${revenue_at_risk*0.3:,.0f}-${revenue_at_risk*0.5:,.0f}.")
    recs.append(f"Monitor {med_risk} medium-risk customers monthly. Implement early-warning triggers when risk score increases above 0.6.")

    # ═══ Q4: SEGMENT ANALYSIS ═══
    cat_cols=[c for c in df.columns if df[c].dtype=='object' and c not in['Customer_ID','Risk_Level'] and df[c].nunique()<15]
    if cat_cols:
        st.markdown('<div class="qs"><div class="qb"><span class="qn">Q4</span><span class="qt">Which segments have the highest churn?</span></div></div>',unsafe_allow_html=True)
        seg_col=cat_cols[0]
        seg_churn=df.groupby(seg_col).agg(Customers=(target_col,'count'),Churned=(target_col,'sum'),Churn_Rate=(target_col,'mean'),Avg_Prob=('Churn_Probability','mean')).round(3)
        seg_churn['Churn_Rate_Pct']=(seg_churn['Churn_Rate']*100).round(1)
        seg_churn=seg_churn.sort_values('Churn_Rate',ascending=False)

        fig_seg=go.Figure(go.Bar(x=seg_churn.index,y=seg_churn['Churn_Rate_Pct'],marker_color=['#ef4444' if v>churn_rate else '#7c3aed' for v in seg_churn['Churn_Rate_Pct']],text=[f'{v}%' for v in seg_churn['Churn_Rate_Pct']],textposition='outside',textfont=dict(size=12,color='#475569')))
        fig_seg.update_layout(title=f'Churn rate by {seg_col.replace("_"," ")}',xaxis_title='',yaxis_title='Churn rate %')
        fig_seg.add_hline(y=churn_rate,line_dash="dot",line_color="#64748b",annotation_text=f"Average: {churn_rate:.1f}%")
        st.markdown('<div class="cw">',unsafe_allow_html=True);st.plotly_chart(sfig(fig_seg),use_container_width=True);st.markdown('</div>',unsafe_allow_html=True)

        worst_seg=seg_churn.index[0]
        worst_rate=seg_churn['Churn_Rate_Pct'].iloc[0]
        best_seg=seg_churn.index[-1]
        best_rate=seg_churn['Churn_Rate_Pct'].iloc[-1]
        st.markdown(f'<div class="co risk"><div class="co-l">Segment insight</div><b>{worst_seg}</b> has the highest churn at <b>{worst_rate}%</b> — {worst_rate/churn_rate:.1f}x the average. <b>{best_seg}</b> has the lowest at <b>{best_rate}%</b>. Investigate what makes {best_seg} customers stick and replicate those conditions for {worst_seg}.</div>',unsafe_allow_html=True)
        recs.append(f"Investigate {worst_seg} segment: churn rate {worst_rate}% is {worst_rate/churn_rate:.1f}x average. Consider targeted offers or product improvements for this group.")

    # ═══ RECOMMENDATIONS ═══
    recs.append("Set up automated monthly churn scoring. Re-train model quarterly with new data for improved accuracy.")
    recs.append("Share this report with Customer Success team to prioritize high-risk accounts for proactive outreach.")

    st.markdown('<div class="qs"><div class="qb"><span class="qn">→</span><span class="qt">Recommended retention actions</span></div></div>',unsafe_allow_html=True)
    items=''.join([f'<div class="ai"><span class="ai-n">{i+1}.</span><span>{r}</span></div>' for i,r in enumerate(recs)])
    st.markdown(f'<div class="ab"><h4>◇ Action items</h4>{items}</div>',unsafe_allow_html=True)

    # Data & Export
    with st.expander("View full customer risk scores"):
        st.dataframe(df[['Customer_ID','Churn_Probability','Risk_Level']+[c for c in display_cols if c not in['Customer_ID','Churn_Probability']]].sort_values('Churn_Probability',ascending=False).head(100) if 'Customer_ID' in df.columns else df.head(100),use_container_width=True)

    st.markdown("")
    d1,d2=st.columns(2)
    risk_export=df[['Customer_ID','Churn_Probability','Risk_Level']].sort_values('Churn_Probability',ascending=False) if 'Customer_ID' in df.columns else None
    d1.download_button("Download full report",data=make_excel(df,{'Best Model':best_name,**best_results},risk_export,recs),file_name=f"churn_report_{datetime.now().strftime('%Y%m%d')}.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    d2.download_button("Download risk scores CSV",data=df.to_csv(index=False).encode('utf-8'),file_name=f"risk_scores_{datetime.now().strftime('%Y%m%d')}.csv",mime="text/csv")

    st.markdown('<div class="af">ML Churn Predictor · Customer Retention Intelligence<br>Random Forest & Logistic Regression · Scikit-learn · All data processed locally</div>',unsafe_allow_html=True)
