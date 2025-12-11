# 🏦 Predicción de Abandono de Clientes Bancarios

## 📹 Demo del Proyecto

https://github.com/nursena8/BankCustomerProject/assets/115145369/Video%20de%20WhatsApp%202025-12-11%20a%20las%2016.31.26_6b7db65f.mp4

---

## 📋 Descripción

Proyecto de machine learning para predecir qué clientes pueden abandonar el banco. Utiliza datos de comportamiento para ayudar al banco a retener clientes y mejorar sus servicios.

## 🎯 Objetivos

- Analizar comportamientos de clientes
- Predecir abandono de clientes
- Optimizar modelos de predicción
- Monitorear rendimiento con Evident AI

## 🚀 Aplicación en Vivo

**[Ver aplicación](https://bankcustomerproject-4feyynfr99jzzpcegoeuq7.streamlit.app/)**

⚠️ **Nota**: Abre la aplicación en Google Chrome (Safari puede tener problemas)

## 🛠️ Tecnologías

- **Python** - Lenguaje principal
- **Pandas & NumPy** - Análisis de datos
- **Scikit-Learn** - Modelos de ML
- **Streamlit** - Aplicación web
- **Evident AI** - Monitoreo de modelos
- **Matplotlib & Seaborn** - Visualizaciones

## 📁 Archivos Importantes

```
├── app.py                          # Aplicación Streamlit
├── project.ipynb                   # Notebook principal
├── gbm_production_model.joblib     # Modelo guardado
├── evidentlyai_monitoring.ipynb    # Monitoreo
└── assets/
    ├── class_report.html
    ├── report.html
    └── data_quality_report.html
```

## 🔧 Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/nursena8/BankCustomerProject.git
cd BankCustomerProject
```

2. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

3. **Ejecutar la aplicación**
```bash
streamlit run app.py
```

4. **Abrir en el navegador**
```
http://localhost:8501
```

## 🐳 Ejecutar con Docker

```bash
# Construir imagen
docker build -t bank-churn-app .

# Ejecutar contenedor
docker run -p 8501:8501 bank-churn-app
```

## 🤖 Modelos Utilizados

- **Clasificación**: GBM, Random Forest, SVC
- **Clustering**: KMeans para segmentación

## 📊 Monitoreo con Evident AI

```bash
pip install evidently
```

El proyecto incluye dashboards para monitorear:
- Rendimiento del modelo
- Calidad de datos
- Reportes de clasificación

## 🔮 Trabajo Futuro

- Mejora continua del modelo
- Agregar nuevas características
- Implementar modelos avanzados
- Sistema de feedback de clientes

## 📄 Licencia

MIT License

## 👩‍💻 Autora

**Nursena Baykir**

📧 nursenabaykir@gmail.com

---

⭐ Si te gustó el proyecto, ¡dale una estrella en GitHub!







