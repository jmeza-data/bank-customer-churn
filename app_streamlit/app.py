import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingClassifier

# -------------------------------------------------------------------
# Rutas de archivos
# -------------------------------------------------------------------
current_directory = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(current_directory, "Bank Customer Churn Prediction.csv")
class_report_path = os.path.join(current_directory, "class_report.html")
data_quality_report_path = os.path.join(current_directory, "data_quality_report.html")
general_report_path = os.path.join(current_directory, "report.html")

# -------------------------------------------------------------------
# Textos en dos idiomas
# -------------------------------------------------------------------
TEXTS = {
    "en": {
        "app_title": "🎯 Customer Churn Prediction",
        "app_subtitle": "Advanced prediction system with intelligent analysis",
        "sidebar_nav": "Navigation",
        "sidebar_lang": "Language / Idioma",
        "page_prediction": "🔮 Prediction",
        "page_class": "📊 Class Report",
        "page_dataq": "✅ Data Quality Report",
        "page_general": "📈 General Report",
        "credit_score": "Credit Score (300–850)",
        "tenure": "Years as Customer",
        "age": "Age",
        "balance": "Account Balance",
        "salary": "Estimated Salary",
        "products": "Number of Products",
        "predict_btn": "🚀 Analyze Customer",
        "prediction_title": "Prediction Results",
        "analysis_title": "📊 Intelligent Analysis",
        "stay_msg": "Low Risk - Customer Likely to Stay",
        "leave_msg": "High Risk - Customer Likely to Leave",
        "error_msg": "Error processing prediction.",
        "prob_label": "Churn Probability",
        "analyzing": "Analyzing customer...",
        "class_title": "Classification Report",
        "dataq_title": "Data Quality Report",
        "general_title": "General Report",
        "customer_profile": "👤 Customer Profile",
        "financial_health": "💰 Financial Health",
        "engagement_level": "📊 Engagement Level",
    },
    "es": {
        "app_title": "🎯 Predicción de Deserción de Clientes",
        "app_subtitle": "Sistema avanzado de predicción con análisis inteligente",
        "sidebar_nav": "Navegación",
        "sidebar_lang": "Idioma / Language",
        "page_prediction": "🔮 Predicción",
        "page_class": "📊 Reporte de Clases",
        "page_dataq": "✅ Reporte de Calidad",
        "page_general": "📈 Reporte General",
        "credit_score": "Puntaje Crediticio (300–850)",
        "tenure": "Años como Cliente",
        "age": "Edad",
        "balance": "Saldo de la Cuenta",
        "salary": "Salario Estimado",
        "products": "Número de Productos",
        "predict_btn": "🚀 Analizar Cliente",
        "prediction_title": "Resultados de Predicción",
        "analysis_title": "📊 Análisis Inteligente",
        "stay_msg": "Riesgo Bajo - Cliente Probablemente Permanecerá",
        "leave_msg": "Riesgo Alto - Cliente Probablemente se Irá",
        "error_msg": "Error al procesar la predicción.",
        "prob_label": "Probabilidad de Deserción",
        "analyzing": "Analizando cliente...",
        "class_title": "Reporte de Clasificación",
        "dataq_title": "Reporte de Calidad de Datos",
        "general_title": "Reporte General",
        "customer_profile": "👤 Perfil del Cliente",
        "financial_health": "💰 Salud Financiera",
        "engagement_level": "📊 Nivel de Compromiso",
    },
}

# -------------------------------------------------------------------
# Funciones auxiliares
# -------------------------------------------------------------------
def load_data():
    data = pd.read_csv(data_path)
    return data


def get_smart_analysis(customer_data, prediction, probability, lang_code):
    """
    Análisis inteligente SIN usar IA (basado en reglas)
    """
    if lang_code == "es":
        analysis = "## 📊 ANÁLISIS DEL RIESGO\n\n"
        
        # Análisis del riesgo
        if probability > 0.7:
            analysis += f"Este cliente presenta un **riesgo muy alto de deserción** ({probability*100:.1f}%). "
        elif probability > 0.5:
            analysis += f"Este cliente presenta un **riesgo moderado-alto de deserción** ({probability*100:.1f}%). "
        elif probability > 0.3:
            analysis += f"Este cliente presenta un **riesgo moderado de deserción** ({probability*100:.1f}%). "
        else:
            analysis += f"Este cliente presenta un **riesgo bajo de deserción** ({probability*100:.1f}%). "
        
        # Factores principales
        factors = []
        
        if customer_data['credit_score'] < 600:
            factors.append("⚠️ **Puntaje crediticio bajo**: Indica posibles problemas financieros")
        elif customer_data['credit_score'] >= 750:
            factors.append("✅ **Excelente puntaje crediticio**: Factor positivo para la retención")
            
        if customer_data['tenure'] < 2:
            factors.append("⚠️ **Cliente nuevo**: Menos de 2 años con el banco, mayor riesgo de deserción")
        elif customer_data['tenure'] >= 5:
            factors.append("✅ **Cliente leal**: Más de 5 años con el banco, buena señal")
            
        if customer_data['products'] <= 1:
            factors.append("⚠️ **Bajo compromiso**: Solo usa 1 producto bancario")
        elif customer_data['products'] >= 3:
            factors.append("✅ **Alto compromiso**: Usa múltiples productos del banco")
            
        if customer_data['balance'] == 0:
            factors.append("⚠️ **Saldo cero**: Cuenta sin movimiento, posible inactividad")
        elif customer_data['balance'] > 100000:
            factors.append("✅ **Alto saldo**: Cliente con recursos significativos")
            
        if customer_data['age'] < 30:
            factors.append("ℹ️ **Cliente joven**: Mayor movilidad entre bancos")
        elif customer_data['age'] > 60:
            factors.append("ℹ️ **Cliente mayor**: Tiende a ser más estable")
        
        analysis += "\n\n## ⚠️ FACTORES CLAVE\n\n"
        for factor in factors[:4]:  # Top 4 factores
            analysis += f"• {factor}\n"
        
        # Recomendaciones
        analysis += "\n\n## 💡 RECOMENDACIONES\n\n"
        
        if probability > 0.6:
            analysis += "• **Acción inmediata**: Contactar al cliente en las próximas 48 horas\n"
            analysis += "• **Oferta personalizada**: Proponer beneficios exclusivos o descuentos\n"
            if customer_data['products'] <= 1:
                analysis += "• **Cross-selling**: Ofrecer productos complementarios con condiciones preferenciales\n"
            analysis += "• **Programa de lealtad**: Inscribir en programa VIP con beneficios especiales\n"
        elif probability > 0.3:
            analysis += "• **Seguimiento proactivo**: Revisar satisfacción del cliente mensualmente\n"
            analysis += "• **Mejorar engagement**: Enviar comunicaciones personalizadas sobre nuevos servicios\n"
            if customer_data['balance'] > 50000:
                analysis += "• **Asesoría financiera**: Ofrecer consultoría de inversión gratuita\n"
            analysis += "• **Incentivos**: Considerar cashback o puntos por uso de productos\n"
        else:
            analysis += "• **Mantener satisfacción**: Continuar con el servicio de calidad actual\n"
            analysis += "• **Oportunidades de crecimiento**: Explorar necesidades adicionales del cliente\n"
            analysis += "• **Programa de referidos**: Incentivar que recomiende el banco a conocidos\n"
            analysis += "• **Comunicación regular**: Mantener contacto para fortalecer la relación\n"
        
        return analysis
    
    else:  # English
        analysis = "## 📊 RISK ANALYSIS\n\n"
        
        if probability > 0.7:
            analysis += f"This customer shows **very high churn risk** ({probability*100:.1f}%). "
        elif probability > 0.5:
            analysis += f"This customer shows **moderate-high churn risk** ({probability*100:.1f}%). "
        elif probability > 0.3:
            analysis += f"This customer shows **moderate churn risk** ({probability*100:.1f}%). "
        else:
            analysis += f"This customer shows **low churn risk** ({probability*100:.1f}%). "
        
        factors = []
        
        if customer_data['credit_score'] < 600:
            factors.append("⚠️ **Low credit score**: Indicates possible financial issues")
        elif customer_data['credit_score'] >= 750:
            factors.append("✅ **Excellent credit score**: Positive factor for retention")
            
        if customer_data['tenure'] < 2:
            factors.append("⚠️ **New customer**: Less than 2 years with bank, higher churn risk")
        elif customer_data['tenure'] >= 5:
            factors.append("✅ **Loyal customer**: Over 5 years with bank, good sign")
            
        if customer_data['products'] <= 1:
            factors.append("⚠️ **Low engagement**: Only using 1 bank product")
        elif customer_data['products'] >= 3:
            factors.append("✅ **High engagement**: Using multiple bank products")
            
        if customer_data['balance'] == 0:
            factors.append("⚠️ **Zero balance**: Account without movement, possible inactivity")
        elif customer_data['balance'] > 100000:
            factors.append("✅ **High balance**: Customer with significant resources")
        
        analysis += "\n\n## ⚠️ KEY FACTORS\n\n"
        for factor in factors[:4]:
            analysis += f"• {factor}\n"
        
        analysis += "\n\n## 💡 RECOMMENDATIONS\n\n"
        
        if probability > 0.6:
            analysis += "• **Immediate action**: Contact customer within 48 hours\n"
            analysis += "• **Personalized offer**: Propose exclusive benefits or discounts\n"
            if customer_data['products'] <= 1:
                analysis += "• **Cross-selling**: Offer complementary products with preferential conditions\n"
            analysis += "• **Loyalty program**: Enroll in VIP program with special benefits\n"
        elif probability > 0.3:
            analysis += "• **Proactive follow-up**: Review customer satisfaction monthly\n"
            analysis += "• **Improve engagement**: Send personalized communications about new services\n"
            if customer_data['balance'] > 50000:
                analysis += "• **Financial advisory**: Offer free investment consultation\n"
            analysis += "• **Incentives**: Consider cashback or points for product usage\n"
        else:
            analysis += "• **Maintain satisfaction**: Continue with current quality service\n"
            analysis += "• **Growth opportunities**: Explore additional customer needs\n"
            analysis += "• **Referral program**: Incentivize recommending bank to acquaintances\n"
            analysis += "• **Regular communication**: Maintain contact to strengthen relationship\n"
        
        return analysis


def calculate_risk_level(probability):
    """Calcula el nivel de riesgo basado en la probabilidad"""
    if probability < 0.3:
        return "bajo", "🟢"
    elif probability < 0.6:
        return "medio", "🟡"
    else:
        return "alto", "🔴"


def get_customer_insights(customer_data, lang_code):
    """Genera insights sobre el perfil del cliente"""
    insights = []
    
    if lang_code == "es":
        if customer_data['credit_score'] >= 750:
            insights.append("💎 Excelente historial crediticio")
        elif customer_data['credit_score'] >= 650:
            insights.append("✅ Buen historial crediticio")
        else:
            insights.append("⚠️ Historial crediticio mejorable")
        
        if customer_data['tenure'] >= 5:
            insights.append("🏆 Cliente de larga data")
        elif customer_data['tenure'] >= 2:
            insights.append("👍 Cliente establecido")
        else:
            insights.append("🆕 Cliente relativamente nuevo")
        
        if customer_data['products'] >= 3:
            insights.append("🎯 Alto compromiso")
        elif customer_data['products'] == 2:
            insights.append("📊 Compromiso moderado")
        else:
            insights.append("📉 Bajo uso de productos")
    else:
        if customer_data['credit_score'] >= 750:
            insights.append("💎 Excellent credit history")
        elif customer_data['credit_score'] >= 650:
            insights.append("✅ Good credit history")
        else:
            insights.append("⚠️ Credit history needs improvement")
        
        if customer_data['tenure'] >= 5:
            insights.append("🏆 Long-term customer")
        elif customer_data['tenure'] >= 2:
            insights.append("👍 Established customer")
        else:
            insights.append("🆕 Relatively new customer")
        
        if customer_data['products'] >= 3:
            insights.append("🎯 High engagement")
        elif customer_data['products'] == 2:
            insights.append("📊 Moderate engagement")
        else:
            insights.append("📉 Low product usage")
    
    return insights


# -------------------------------------------------------------------
# Configuración de la página
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction", 
    page_icon="🎯", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stMetric label {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: white !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: white !important;
    }
    .insight-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar session state
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False

# Selección de idioma
lang = st.sidebar.selectbox("🌐 Language / Idioma", ["Español", "English"])
lang_code = "es" if lang == "Español" else "en"
T = TEXTS[lang_code]

# Navegación
st.sidebar.markdown("---")
st.sidebar.markdown(f"## {T['sidebar_nav']}")

pages = {
    "prediction": T["page_prediction"],
    "class": T["page_class"],
    "dataq": T["page_dataq"],
    "general": T["page_general"],
}

page_key = st.sidebar.radio(
    "",
    list(pages.keys()),
    format_func=lambda k: pages[k],
)

st.sidebar.markdown("---")
st.sidebar.success("💡 **Análisis inteligente completo**\n\nSin límites de uso" if lang_code == "es" 
                else "💡 **Complete intelligent analysis**\n\nNo usage limits")

# -------------------------------------------------------------------
# Página: Predicción
# -------------------------------------------------------------------
if page_key == "prediction":
    st.title(T["app_title"])
    st.markdown(f"*{T['app_subtitle']}*")
    st.markdown("---")

    data = load_data()

    # Formulario de entrada
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"### {T['customer_profile']}")
        credit_score = st.number_input(
            T["credit_score"], 
            min_value=300, 
            max_value=850, 
            value=750,
            step=10,
            key="credit_score_input"
        )
        age = st.slider(T["age"], 18, 92, 30, key="age_input")
        tenure = st.selectbox(
            T["tenure"],
            list(range(1, 21)),
            index=0,
            key="tenure_input"
        )
    
    with col2:
        st.markdown(f"### {T['financial_health']}")
        balance = st.number_input(
            T["balance"], 
            min_value=0.0, 
            max_value=300000.0, 
            value=50000.0,
            step=1000.0,
            format="%.2f",
            key="balance_input"
        )
        estimated_salary = st.number_input(
            T["salary"], 
            min_value=0.0, 
            max_value=300000.0, 
            value=75000.0,
            step=1000.0,
            format="%.2f",
            key="salary_input"
        )
    
    with col3:
        st.markdown(f"### {T['engagement_level']}")
        products_number = st.slider(T["products"], 0, 4, 1, key="products_input")
        st.markdown("")
        st.markdown("")
        predict_clicked = st.button(T["predict_btn"], use_container_width=True, type="primary")

    # Procesar predicción
    if predict_clicked:
        with st.spinner(T["analyzing"]):
            # Preparar datos
            customer_data = {
                'credit_score': float(credit_score),
                'tenure': float(tenure),
                'age': float(age),
                'balance': float(balance),
                'salary': float(estimated_salary),
                'products': float(products_number),
            }
            
            row = np.array([
                customer_data['credit_score'],
                customer_data['tenure'],
                customer_data['age'],
                customer_data['balance'],
                customer_data['salary'],
                customer_data['products'],
            ])

            # Entrenar modelo
            X_train = data[
                ["credit_score", "tenure", "age", "balance", "estimated_salary", "products_number"]
            ]
            y_train = data["churn"]

            model = GradientBoostingClassifier(
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=2,
                n_estimators=100,
                random_state=42
            )
            model.fit(X_train, y_train)

            pred = model.predict([row])[0]
            prob = model.predict_proba([row])[0][1]
            
            risk_level, risk_icon = calculate_risk_level(prob)
            insights = get_customer_insights(customer_data, lang_code)
            smart_analysis = get_smart_analysis(customer_data, pred, prob, lang_code)
            feature_importance = model.feature_importances_
            
            # Guardar en session state
            st.session_state.prediction_done = True
            st.session_state.current_prediction = {
                'customer_data': customer_data,
                'pred': pred,
                'prob': prob,
                'risk_level': risk_level,
                'risk_icon': risk_icon,
                'insights': insights,
                'smart_analysis': smart_analysis,
                'feature_importance': feature_importance
            }

    # Mostrar resultados si existen
    if st.session_state.prediction_done:
        # Recuperar datos
        data_pred = st.session_state.current_prediction
        customer_data = data_pred['customer_data']
        pred = data_pred['pred']
        prob = data_pred['prob']
        risk_level = data_pred['risk_level']
        risk_icon = data_pred['risk_icon']
        insights = data_pred['insights']
        smart_analysis = data_pred['smart_analysis']
        feature_importance = data_pred['feature_importance']
        
        # Resultados visuales
        st.markdown("## " + T["prediction_title"])
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric(
                label=T["prob_label"], 
                value=f"{prob*100:.1f}%",
                delta=f"{risk_icon} {risk_level.upper()}"
            )
        
        with col_b:
            retention_prob = (1 - prob) * 100
            st.metric(
                label="Prob. Retención" if lang_code == "es" else "Retention Prob.",
                value=f"{retention_prob:.1f}%"
            )
        
        with col_c:
            if pred == 1:
                st.error(T["leave_msg"])
            else:
                st.success(T["stay_msg"])
        
        st.markdown("---")
        
        # Insights del cliente
        st.markdown("### 📌 Insights del Perfil" if lang_code == "es" else "### 📌 Profile Insights")
        
        insight_cols = st.columns(len(insights))
        for idx, text in enumerate(insights):
            with insight_cols[idx]:
                st.info(text)
        
        st.markdown("---")
        
        # Análisis inteligente (SIN IA - SIEMPRE FUNCIONA)
        st.markdown(f"## {T['analysis_title']}")
        
        st.markdown(f"""
        <div class="insight-card">
        
{smart_analysis}
        
        </div>
        """, unsafe_allow_html=True)
        
        # Feature importance
        st.markdown("---")
        st.markdown("### 📊 Factores de Influencia" if lang_code == "es" else "### 📊 Influence Factors")
        
        features = ["Puntaje" if lang_code == "es" else "Score", 
                   "Antigüedad" if lang_code == "es" else "Tenure", 
                   "Edad" if lang_code == "es" else "Age", 
                   "Saldo" if lang_code == "es" else "Balance", 
                   "Salario" if lang_code == "es" else "Salary", 
                   "Productos" if lang_code == "es" else "Products"]
        
        importance_df = pd.DataFrame({
            'Factor': features,
            'Importancia': feature_importance * 100
        }).sort_values('Importancia', ascending=False)
        
        st.bar_chart(importance_df.set_index('Factor'))

# -------------------------------------------------------------------
# Otras páginas
# -------------------------------------------------------------------
elif page_key == "class":
    st.markdown(f"# {T['class_title']}")
    try:
        with open(class_report_path, "r", encoding="utf-8", errors="ignore") as file:
            class_html_content = file.read()
        st.components.v1.html(class_html_content, height=3000, scrolling=True)
    except FileNotFoundError:
        st.error("Archivo no encontrado" if lang_code == "es" else "File not found")

elif page_key == "dataq":
    st.markdown(f"# {T['dataq_title']}")
    try:
        with open(data_quality_report_path, "r", encoding="utf-8", errors="ignore") as file:
            data_quality_html_content = file.read()
        st.components.v1.html(data_quality_html_content, height=800, scrolling=True)
    except FileNotFoundError:
        st.error("Archivo no encontrado" if lang_code == "es" else "File not found")

elif page_key == "general":
    st.markdown(f"# {T['general_title']}")
    try:
        with open(general_report_path, "r", encoding="utf-8", errors="ignore") as file:
            general_html_content = file.read()
        st.components.v1.html(general_html_content, height=2000, scrolling=True)
    except FileNotFoundError:
        st.error("Archivo no encontrado" if lang_code == "es" else "File not found")