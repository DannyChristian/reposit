from flask import Flask, request, render_template_string
import pandas as pd
import plotly.express as px
import io

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Dengue Perú - Series de Tiempo</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; background-color: #f8f9fa; }
        iframe { width: 100%; height: 500px; border: 1px solid #ddd; border-radius: 8px; background: white; }
        table { margin-top: 20px; }
        th, td { text-align: center; }
    </style>
</head>
<body>
<div class="container">
    <h1 class="text-center mb-4">Reconocimiento de Patrones - Dengue Perú</h1>
    <form method="post" enctype="multipart/form-data">
        <div class="mb-3">
            <label for="csvfile" class="form-label">Sube tu archivo CSV</label>
            <input class="form-control" type="file" name="csvfile" required>
        </div>
        <button class="btn btn-primary" type="submit">Cargar y Analizar</button>
    </form>

    {% if stats %}
        <hr>
        <h3>Resumen Estadístico</h3>
        <p><strong>Total de casos:</strong> {{ stats.total_cases }}</p>
        <div class="row">
            <div class="col-md-6">
                <h5>Casos por Año</h5>
                <ul>
                    {% for year, count in stats.cases_by_year.items() %}
                        <li><strong>{{ year }}:</strong> {{ count }}</li>
                    {% endfor %}
                </ul>
            </div>
            <div class="col-md-6">
                <h5>Top 5 Departamentos</h5>
                <ul>
                    {% for dep, count in stats.top_departments.items() %}
                        <li><strong>{{ dep }}:</strong> {{ count }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>

        <hr>
        <h3>Casos Semanales por Año</h3>
        {{ trends_plot|safe }}

        <h3>Casos Totales por Departamento</h3>
{{ total_by_department_plot|safe }}


        <h3>Casos Mensuales por Año</h3>
        {{ monthly_plot|safe }}

        <hr>
        <h3>Tabla I - Distribución por Edad, Severidad y Género</h3>
        {{ table1|safe }}

        <h3>Tabla II - Casos Mensuales de Dengue en Perú (2020–2023)</h3>
        {{ table2|safe }}

        <h3>Tabla III - Casos por Departamento, Año y Gravedad</h3>
        {{ table3|safe }}
    {% endif %}
</div>
</body>
</html>
'''

def process_csv(file):
    df = pd.read_csv(file, sep=None, engine='python')
    df.columns = df.columns.str.strip().str.lower()
    df = df[df['ano'].between(2020, 2023)]
    if 'mes' not in df.columns:
        df['mes'] = ((df['semana'] - 1) // 4 + 1).clip(upper=12)
    return df

def get_summary(df):
    return {
        'total_cases': len(df),
        'cases_by_year': df['ano'].value_counts().sort_index().to_dict(),
        'top_departments': df['departamento'].value_counts().head(5).to_dict()
    }

def plot_trends(df):
    data = df.groupby(['ano', 'semana']).size().reset_index(name='cases')
    fig = px.line(data, x='semana', y='cases', color='ano', markers=True, title='Casos semanales de dengue')
    fig.update_layout(template='plotly_white')
    return fig.to_html(full_html=False)

def plot_heatmap(df):
    data = df.groupby(['departamento', 'semana']).size().reset_index(name='cases')
    fig = px.density_heatmap(data, x='cases', y='departamento', z='cases', color_continuous_scale='Reds', title='Casos totales por Departamento')
    fig.update_layout(template='plotly_white')
    return fig.to_html(full_html=False)

def plot_monthly(df):
    data = df.groupby(['ano', 'mes']).size().reset_index(name='cases')
    fig = px.bar(data, x='mes', y='cases', color='ano', barmode='group', title='Casos mensuales por año')
    fig.update_layout(template='plotly_white')
    return fig.to_html(full_html=False)

def plot_total_cases_by_department(df):
    data = df['departamento'].value_counts().reset_index()
    data.columns = ['departamento', 'cases']
    fig = px.bar(data.sort_values('cases'), 
                 x='cases', y='departamento', 
                 orientation='h',
                 color='cases', 
                 color_continuous_scale='Reds',
                 title='Casos totales por Departamento')
    fig.update_layout(template='plotly_white', yaxis_title='Departamento', xaxis_title='Casos')
    return fig.to_html(full_html=False)


def generate_tables(df):
    # Tabla I - Distribución por Edad, Severidad y Género
    df['grave'] = df['diagnostic'].str.contains("CON SIGNOS|GRAVE", case=False)
    df['grupo_edad'] = pd.cut(df['edad'], bins=[-1,0,4,14,29,59,150],
                              labels=["0–11 meses","1–4 años","5–14 años","15–29 años","30–59 años","60+ años"])
    tab1 = df.groupby('grupo_edad').agg(
        Totales=('edad', 'count'),
        Graves=('grave', 'sum'),
        F=('sexo', lambda x: (x == 'F').sum()),
        M=('sexo', lambda x: (x == 'M').sum())
    )
    tab1['%'] = (tab1['Totales'] / tab1['Totales'].sum() * 100).round(2).astype(str) + '%'
    table1 = tab1.reset_index().to_html(index=False, classes="table table-bordered table-striped")

    # Tabla II - Casos mensuales
    tab2 = df.groupby(['mes', 'ano']).size().unstack().fillna(0).astype(int)
    tab2.index = ['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre']
    table2 = tab2.reset_index().rename(columns={'index': 'Mes'}).to_html(index=False, classes="table table-bordered table-striped")

    # Tabla III - Casos por departamento, año y gravedad
    tab3 = df.groupby(['departamento', 'ano']).size().unstack().fillna(0).astype(int)
    tab3['Total'] = tab3.sum(axis=1)
    graves = df[df['grave']].groupby('departamento').size()
    tab3['Graves'] = graves
    table3 = tab3.reset_index().to_html(index=False, classes="table table-bordered table-striped")

    return table1, table2, table3

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['csvfile']
        if not file:
            return render_template_string(HTML_TEMPLATE, stats=None)
        df = process_csv(file)
        stats = get_summary(df)
        trends_plot = plot_trends(df)
        heatmap_plot = plot_heatmap(df)
        monthly_plot = plot_monthly(df)
        total_by_department_plot = plot_total_cases_by_department(df)

        table1, table2, table3 = generate_tables(df)
        return render_template_string(HTML_TEMPLATE,
    stats=stats,
    trends_plot=trends_plot,
    heatmap_plot=heatmap_plot,
    monthly_plot=monthly_plot,
    total_by_department_plot=total_by_department_plot,
    table1=table1,
    table2=table2,
    table3=table3
)

    return render_template_string(HTML_TEMPLATE, stats=None)

if __name__ == '__main__':
    app.run(debug=True)
