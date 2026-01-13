# Análisis de Riesgo de Pobreza en España

## Encuesta de Condiciones de Vida (ECV) 2024 - INE

### Descripción del conjunto de datos

Este conjunto de datos proviene de la **Encuesta de Condiciones de Vida (ECV) 2024** del Instituto Nacional de Estadística (INE) de España. La ECV es la versión española de la encuesta EU-SILC (European Union Statistics on Income and Living Conditions) y proporciona información detallada sobre ingresos, condiciones de vida y riesgo de pobreza en hogares españoles.

**Objetivo del proyecto:** Desarrollar un modelo de regresión para predecir el riesgo de pobreza de personas/hogares en España utilizando datos demográficos y socioeconómicos.

---

## Estructura de los Datos

El conjunto de datos se compone de **4 ficheros transversales** (Base 2013):

### 1. **ECV_Td_2024** - Datos Básicos del Hogar

- **Registros:** 29,781 hogares
- **Contenido:** Variables básicas a nivel de hogar
- **Formato:** CSV/TAB en `data/ECV_Td_2024/CSV/`

### 2. **ECV_Tr_2024** - Datos Básicos de la Persona

- **Contenido:** Variables básicas a nivel individual
- **Formato:** CSV/TAB en `data/ECV_Tr_2024/CSV/`

### 3. **ECV_Th_2024** - Datos Detallados del Hogar ⭐

- **Registros:** 29,781 hogares
- **Contenido:** Variables detalladas del hogar incluyendo **riesgo de pobreza** (variable objetivo)
- **Archivo principal:** `data/ECV_Th_2024/CSV/ECV_Th_2024.tab`
- **Formato:** TSV (campos separados por tabulador)

### 4. **ECV_Tp_2024** - Datos Detallados de los Adultos ⭐

- **Registros:** 61,526 adultos
- **Contenido:** Variables demográficas, educación, empleo, salud, ingresos individuales
- **Archivo principal:** `data/ECV_Tp_2024/CSV/ECV_Tp_2024.tab`
- **Formato:** TSV (campos separados por tabulador)

### Documentación Adicional

- **Diseños de registro:** `data/disreg_ecv_24/*.xlsx` (descripciones de variables en Excel)
- **Metadatos:** `data/ECV_T*/md_ECV_T*_2024.txt` (diccionarios de datos detallados)
- **Instrucciones:** `data/LeemeECV_2024.txt`

---

## Variable Objetivo: Riesgo de Pobreza

### `vhPobreza` (Columna 131 en ECV_Th_2024)

- **Ubicación:** Archivo de hogares detallados (`ECV_Th_2024`)
- **Tipo:** Binaria (categórica)
- **Valores:**
  - `0` = No en riesgo de pobreza
  - `1` = Sí en riesgo de pobreza
- **Descripción:** Indica si el hogar está en riesgo de pobreza según los criterios EU-SILC

### Ejemplo de Datos Reales:

```
Hogar 1: Tamaño=5 personas, vhPobreza=1, Ingreso=4,600€
Hogar 2: Tamaño=8 personas, vhPobreza=0, Ingreso=42,420€
Hogar 3: Tamaño=10 personas, vhPobreza=0, Ingreso=32,569€
```

### Variable Complementaria: `vhMATDEP` (Columna 132)

- **Descripción:** Privación material del hogar
- **Valores:** 0 (No) / 1 (Sí)

---

## Variables Predictoras Clave

### A. Variables de Ingresos (Archivo: ECV_Th_2024)

| Variable | Columna | Descripción |
|----------|---------|-------------|
| **HY020** | 16 | Ingreso disponible total del hogar (neto) |
| **HY022** | 18 | Ingreso disponible del hogar antes de transferencias sociales |
| **HY023** | 20 | Ingreso disponible del hogar antes de transferencias sociales excepto pensiones |
| **HY030N** | - | Renta imputada de la vivienda |
| **HY040N** | - | Ingresos por alquiler de propiedades |
| **HY050N** | - | Ingresos familiares/de menores |
| **HY070N** | - | Ingresos por intereses, dividendos, etc. |

### B. Variables Demográficas del Hogar (Archivo: ECV_Th_2024)

| Variable | Columna | Descripción |
|----------|---------|-------------|
| **HB070** | 8 | Número de miembros del hogar (tamaño del hogar) |
| **HB100** | 12 | Número de personas de 0-15 años en el hogar |
| **HB060** | - | Tipo de hogar (unipersonal, pareja con hijos, etc.) |

### C. Variables Demográficas Individuales (Archivo: ECV_Tp_2024)

| Variable | Columna | Descripción | Valores |
|----------|---------|-------------|---------|
| **PB110** | 8 | Año de nacimiento | - |
| **PB140** | 12 | Año de nacimiento | - |
| **PB150** | 14 | Sexo | 1=Hombre, 2=Mujer |
| **PB180** | 20 | País de nacimiento (cónyuge) | - |
| **PB190** | 22 | Estado civil | 1=Soltero, 2=Casado, 3=Separado, 4=Viudo, 5=Divorciado |

### D. Variables de Educación (Archivo: ECV_Tp_2024)

| Variable | Columna | Descripción | Valores |
|----------|---------|-------------|---------|
| **PE021** | 34 | Nivel educativo alcanzado | 00=Menos que primaria<br>10=Primaria<br>20=1ª ESO<br>30=2ª ESO<br>40=Postsecundaria<br>50=Superior |
| **PE041** | - | Nivel educativo detallado (16-34 años) | - |

### E. Variables de Empleo (Archivo: ECV_Tp_2024)

| Variable | Descripción |
|----------|-------------|
| **PL051A** | Situación laboral actual |
| **PL051B** | Situación laboral el año anterior |
| **PL060** | Horas trabajadas por semana |
| **PL073-076** | Meses trabajados a tiempo completo/parcial |
| **PL080** | Ocupación (código ISCO) |

### F. Variables de Ingresos Individuales (Archivo: ECV_Tp_2024)

| Variable | Descripción |
|----------|-------------|
| **PY010N** | Ingresos brutos del trabajo por cuenta ajena |
| **PY020N** | Ingresos brutos del trabajo por cuenta propia |
| **PY050N** | Ingresos por prestaciones de desempleo |
| **PY090N** | Ingresos por pensión de jubilación |
| **PY100N** | Pensión de supervivencia |
| **PY110N** | Pensión por enfermedad/discapacidad |
| **PY120N** | Pensión por viudedad |
| **PY130N** | Ingresos por familia/hijos |
| **PY140N** | Otras prestaciones sociales |

### G. Variables de Salud (Archivo: ECV_Tp_2024)

| Variable | Descripción |
|----------|-------------|
| **PH010** | Estado de salud general (1=Muy bueno a 5=Muy malo) |
| **PH020** | Limitación en actividades por problemas de salud |
| **PH030** | Enfermedad crónica |

### H. Variables de Privación Material (Archivo: ECV_Th_2024)

| Variable | Descripción |
|----------|-------------|
| **HS010-HS190** | Capacidad para hacer frente a gastos imprevistos, ir de vacaciones, pagar la vivienda, etc. |
| **HH010-HH090** | Problemas con la vivienda (humedad, luz, etc.) |

---

## Estrategia para el Análisis de Machine Learning

### 1. **Unión de Datos (Data Merging)**

Para crear un conjunto de datos completo para el modelo, necesitarás unir los archivos:

```python
# Pseudocódigo
hogares = pd.read_csv('ECV_Th_2024.tab', sep='\t')  # Variable objetivo
personas = pd.read_csv('ECV_Tp_2024.tab', sep='\t')

# Unir por ID de hogar (HB030 en hogares, PB030 en personas)
# Opción 1: Agregar datos de personas por hogar
# Opción 2: Usar solo cabeza de familia/sustentador principal
```

**Claves de unión:**
- `HB030` (ECV_Th) ↔ `PB030` (ECV_Tp): ID del hogar
- `DB030` (ECV_Td) ↔ `HB030` (ECV_Th): ID del hogar

### 2. **Features Engineering Recomendadas**

- **Edad:** Calcular a partir de año de nacimiento (2024 - PB110)
- **Ratio ingresos/miembros:** HY020 / HB070 (ingreso per cápita)
- **Tasa de dependencia:** (niños + mayores) / adultos en edad laboral
- **Empleo del hogar:** Número de empleados / total de adultos
- **Nivel educativo máximo del hogar:** Max(PE021) por hogar
- **Ingresos diversificados:** Número de fuentes de ingreso activas
- **Composición familiar:** Variables dummy para tipos de hogar
- **Indicadores de privación:** Suma de variables HS/HH

### 3. **Tipos de Modelos Posibles**

#### Clasificación

Dado que `vhPobreza` es binaria (0/1):
- **Regresión Logística** (baseline interpretable)
- **Random Forest / Gradient Boosting** (XGBoost, LightGBM)
- **SVM con kernel RBF**
- **Redes Neuronales** (si hay suficientes datos)

#### Regresión

Predecir ingresos continuos (HY020) y derivar riesgo:
- **Regresión Lineal** (baseline)
- **Ridge/Lasso/ElasticNet** (con regularización)
- **Regresión con Gradient Boosting**

### 4. **Validación y Métricas**

Para el problema de clasificación:
- **Métrica principal:** F1-Score (conjunto de datos probablemente desbalanceado)
- **Métricas secundarias:** Precision, Recall, AUC-ROC
- **Validación cruzada:** Estratificada (k-fold=5 o 10)
- **Análisis:** Matriz de confusión, curva ROC

### 5. **Consideraciones Importantes**

#### Desbalanceo de Clases

```python
# Verificar distribución de vhPobreza
print(hogares['vhPobreza'].value_counts())

# Si está desbalanceado (probable), usar:
# - SMOTE para sobremuestreo
# - class_weight='balanced' en scikit-learn
# - StratifiedKFold para validación
```

#### Valores Faltantes

Los archivos usan códigos especiales:
- `-1`, `-2`, `-3`, `-4`, `-5`, `-6`: Diferentes tipos de missing data
- Variables con sufijo `_F`: Flags de calidad de datos

```python
# Ejemplo de limpieza
df = df.replace([-1, -2, -3, -4, -5, -6], np.nan)
```

#### Multicolinealidad

- Cuidado con variables de ingresos (HY020, HY022, HY023) altamente correlacionadas
- Usar VIF (Variance Inflation Factor) o eliminar variables redundantes
- Considerar PCA si hay muchas variables correlacionadas

#### Feature Scaling

- Normalizar/estandarizar variables continuas (ingresos, edad, horas trabajadas)
- Dejar variables categóricas como dummy variables

---

## Guía Rápida de Inicio

### 1. Cargar Datos

```python
import pandas as pd

hogares = pd.read_csv(
   'data/ECV_Th_2024/CSV/ECV_Th_2024.tab',
   sep='\t',
   encoding='latin-1'
)

personas = pd.read_csv(
   'data/ECV_Tp_2024/CSV/ECV_Tp_2024.tab',
   sep='\t',
   encoding='latin-1'
)

print(f"Hogares: {hogares.shape}")
print(f"Personas: {personas.shape}")
```

### 2. Explorar Variable Objetivo

```python
# Distribución de riesgo de pobreza
print(hogares['vhPobreza'].value_counts())
print(hogares['vhPobreza'].value_counts(normalize=True))

# Relación con ingresos
hogares.groupby('vhPobreza')['HY020'].describe()
```

### 3. Seleccionar Variables Clave

```python
# Variables de hogar
vars_hogar = ['HB030', 'HB070', 'HB100', 'HY020', 'HY022', 'vhPobreza']
hogares_subset = hogares[vars_hogar]

# Variables de personas (cabeza de familia)
vars_persona = ['PB030', 'PB150', 'PE021', 'PL051A', 'PY010N']
personas_subset = personas[vars_persona]
```

### 4. Pipeline Básico

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

# 1. Preprocesar y unir datos
# 2. Manejar valores faltantes
# 3. Codificar variables categóricas
# 4. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Entrenar modelo
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 6. Evaluar
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## Preguntas de Investigación Sugeridas

1. **¿Cuáles son los factores demográficos más predictivos del riesgo de pobreza?**
   - Edad, sexo, estado civil, composición del hogar

2. **¿Cómo influye la educación en el riesgo de pobreza?**
   - Comparar niveles educativos (PE021) con vhPobreza

3. **¿Qué peso tiene el empleo vs. las pensiones/subsidios?**
   - Analizar fuentes de ingreso (PY010N vs PY090N, PY050N)

4. **¿Existe un umbral de ingreso claro para el riesgo de pobreza?**
   - Distribución de HY020 por vhPobreza

5. **¿El tamaño del hogar es un factor de riesgo significativo?**
   - Relación entre HB070 y vhPobreza controlando por ingresos

6. **¿La privación material (vhMATDEP) es un predictor independiente?**
   - Correlación entre vhMATDEP y vhPobreza

---

## Recursos Adicionales

- **Web del INE:** https://www.ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736176807&menu=resultados&idp=1254735976608
- **Metodología EU-SILC:** Consultar documentación europea sobre indicadores de pobreza y exclusión social
- **Variables flag (_F):** Indican calidad/validez del dato (11=bueno, otros códigos=problemas)

---

## Notas Técnicas

- **Encoding:** Los archivos CSV usan `latin-1` o `ISO-8859-1` (caracteres españoles con tildes)
- **Separador:** TAB (`\t`) en archivos `.tab`
- **Pesos muestrales:** Algunas variables incluyen factores de elevación para representatividad poblacional
- **Año de referencia:** Los datos de 2024 pueden referirse a ingresos del año anterior (2023)

---

## Estructura de Directorios

```
data/
├── LeemeECV_2024.txt                    # Instrucciones generales
├── disreg_ecv_24/                       # Diseños de registro (Excel)
│   ├── dr_ECV_CM_Td_2024.xlsx           # Descripción variables Td
│   ├── dr_ECV_CM_Th_2024.xlsx           # Descripción variables Th
│   ├── dr_ECV_CM_Tp_2024.xlsx           # Descripción variables Tp
│   └── dr_ECV_CM_Tr_2024.xlsx           # Descripción variables Tr
├── ECV_Td_2024/
│   ├── CSV/
│   │   ├── ECV_Td_2024.tab              # Datos básicos hogar (TSV)
│   │   └── esudb24d.csv                 # Formato alternativo (CSV)
│   └── md_ECV_Td_2024.txt               # Metadatos (diccionario)
├── ECV_Th_2024/                         # Variable objetivo
│   ├── CSV/
│   │   ├── ECV_Th_2024.tab              # Datos detallados hogar (TSV)
│   │   └── esudb24h.csv                 # Formato alternativo (CSV)
│   └── md_ECV_Th_2024.txt               # Metadatos (diccionario)
├── ECV_Tp_2024/                         # Variables demográficas
│   ├── CSV/
│   │   ├── ECV_Tp_2024.tab              # Datos detallados personas (TSV)
│   │   └── esudb24p.csv                 # Formato alternativo (CSV)
│   └── md_ECV_Tp_2024.txt               # Metadatos (diccionario)
└── ECV_Tr_2024/
    ├── CSV/
    │   ├── ECV_Tr_2024.tab              # Datos básicos persona (TSV)
    │   └── esudb24r.csv                 # Formato alternativo (CSV)
    └── md_ECV_Tr_2024.txt               # Metadatos (diccionario)
```

---

**Creado para:** Proyecto de Machine Learning con Scikit-Learn
**Conjunto de datos:** Encuesta de Condiciones de Vida (ECV) 2024 - INE España
**Fecha:** Enero 2024
