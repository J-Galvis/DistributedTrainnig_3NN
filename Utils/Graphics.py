import numpy as np
import pandas as pd
import altair as alt

def graficar_resultados(historial_loss, historial_acc):
    """
    Grafica la pérdida y la precisión durante el entrenamiento.
    Requiere altair, que es solo para visualización, no para la red.
    """
    try:
        epocas = range(1, len(historial_loss) + 1)
        
        # Crear DataFrame con los datos
        df = pd.DataFrame({
            'Época': list(epocas),
            'Loss': historial_loss,
            'Precisión': historial_acc
        })

        # Gráfica de pérdida
        loss_chart = alt.Chart(df).mark_line(color='#1f77b4', size=2).encode(
            x=alt.X('Época:Q', title='Época'),
            y=alt.Y('Loss:Q', title='Cross-Entropy Loss')
        ).properties(
            title='Pérdida (Loss) durante el entrenamiento',
            width=500,
            height=300
        )

        # Gráfica de precisión
        acc_chart = alt.Chart(df).mark_line(color='#2ca02c', size=2).encode(
            x=alt.X('Época:Q', title='Época'),
            y=alt.Y('Precisión:Q', title='Precisión (%)', scale=alt.Scale(domain=[0, 100]))
        ).properties(
            title='Precisión (Train Accuracy) durante el entrenamiento',
            width=500,
            height=300
        )

        # Combinar gráficas
        chart = alt.hconcat(loss_chart, acc_chart)
        
        # Guardar como HTML
        chart.save('entrenamiento.html')
        print("\n  Gráfica guardada en 'entrenamiento.html'")

    except ImportError:
        print("\n  (altair no disponible, se omite la gráfica)")

# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICA EXTENDIDA PARA EL ALGORITMO DE ARNOVI
# ─────────────────────────────────────────────────────────────────────────────

def graficar_arnovi(historiales_loss, historiales_acc,
                    historial_loss_prom, historial_acc_prom,
                    num_particiones):
    """
    Genera gráficas que muestran:
      - La pérdida y precisión de CADA partición (curvas individuales)
      - La pérdida y precisión PROMEDIO (curva gruesa)

    Esto permite visualizar cómo cada mini-red aprende de forma diferente
    y cómo el promedio captura el comportamiento general.
    """
    try:
        epocas = range(1, len(historial_loss_prom) + 1)
        
        # Preparar datos para pérdida
        loss_data = []
        for k in range(num_particiones):
            for e, loss_val in enumerate(historiales_loss[k], 1):
                loss_data.append({
                    'Época': e,
                    'Loss': loss_val,
                    'Tipo': f'Partición {k+1}',
                    'es_promedio': False
                })
        for e, loss_val in enumerate(historial_loss_prom, 1):
            loss_data.append({
                'Época': e,
                'Loss': loss_val,
                'Tipo': 'Promedio',
                'es_promedio': True
            })
        
        df_loss = pd.DataFrame(loss_data)
        
        # Preparar datos para precisión
        acc_data = []
        for k in range(num_particiones):
            for e, acc_val in enumerate(historiales_acc[k], 1):
                acc_data.append({
                    'Época': e,
                    'Precisión': acc_val,
                    'Tipo': f'Partición {k+1}',
                    'es_promedio': False
                })
        for e, acc_val in enumerate(historial_acc_prom, 1):
            acc_data.append({
                'Época': e,
                'Precisión': acc_val,
                'Tipo': 'Promedio',
                'es_promedio': True
            })
        
        df_acc = pd.DataFrame(acc_data)
        
        # Gráfica de pérdida con capas
        loss_particiones = alt.Chart(df_loss[~df_loss['es_promedio']]).mark_line(
            opacity=0.4, size=1
        ).encode(
            x='Época:Q',
            y='Loss:Q',
            color=alt.Color('Tipo:N', legend=alt.Legend(labelFontSize=8))
        )
        
        loss_promedio = alt.Chart(df_loss[df_loss['es_promedio']]).mark_line(
            color='black', size=2.5
        ).encode(
            x='Época:Q',
            y='Loss:Q'
        )
        
        loss_chart = alt.layer(loss_particiones, loss_promedio).properties(
            title='Pérdida (Loss) — Algoritmo de Arnovi',
            width=600,
            height=350
        ).encode(
            x=alt.X('Época:Q', title='Época'),
            y=alt.Y('Loss:Q', title='Cross-Entropy Loss')
        )
        
        # Gráfica de precisión con capas
        acc_particiones = alt.Chart(df_acc[~df_acc['es_promedio']]).mark_line(
            opacity=0.4, size=1
        ).encode(
            x='Época:Q',
            y='Precisión:Q',
            color=alt.Color('Tipo:N', legend=alt.Legend(labelFontSize=8))
        )
        
        acc_promedio = alt.Chart(df_acc[df_acc['es_promedio']]).mark_line(
            color='black', size=2.5
        ).encode(
            x='Época:Q',
            y='Precisión:Q'
        )
        
        acc_chart = alt.layer(acc_particiones, acc_promedio).properties(
            title='Precisión (Train Acc) — Algoritmo de Arnovi',
            width=600,
            height=350
        ).encode(
            x=alt.X('Época:Q', title='Época'),
            y=alt.Y('Precisión:Q', title='Precisión (%)', scale=alt.Scale(domain=[0, 100]))
        )
        
        # Combinar gráficas
        chart = alt.hconcat(loss_chart, acc_chart)
        
        # Guardar como HTML
        chart.save('entrenamiento_arnovi.html')
        print("\n  Gráfica guardada en 'entrenamiento_arnovi.html'")

    except ImportError:
        print("\n  (altair no disponible, se omite la gráfica)")


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICA PARA EL ALGORITMO DE DIEGO
# ─────────────────────────────────────────────────────────────────────────────

def graficar_diego(historial_loss, historial_acc, historial_acc_test,
                   historiales_loss_particiones, historiales_acc_particiones,
                   num_particiones):
    """
    Genera 3 gráficas para el Algoritmo de Diego:
      1. Pérdida: curvas individuales por partición + curva global promediada
      2. Precisión en Train: curvas individuales + curva global
      3. Precisión en Test del modelo promediado por época

    Esto permite ver:
      - Cómo cada partición se comporta independientemente (curvas tenues)
      - El resultado DESPUÉS del promediado en cada época (curva gruesa)
      - La evolución del modelo final en datos nunca vistos (test)
    """
    try:
        epocas = range(1, len(historial_loss) + 1)
        
        # Preparar datos para pérdida
        loss_data = []
        for k in range(num_particiones):
            for e, loss_val in enumerate(historiales_loss_particiones[k], 1):
                loss_data.append({
                    'Época': e,
                    'Loss': loss_val,
                    'Tipo': f'Partición {k+1}',
                    'es_global': False
                })
        for e, loss_val in enumerate(historial_loss, 1):
            loss_data.append({
                'Época': e,
                'Loss': loss_val,
                'Tipo': 'Global (promediado)',
                'es_global': True
            })
        
        df_loss = pd.DataFrame(loss_data)
        
        # Preparar datos para precisión de train
        acc_data = []
        for k in range(num_particiones):
            for e, acc_val in enumerate(historiales_acc_particiones[k], 1):
                acc_data.append({
                    'Época': e,
                    'Precisión': acc_val,
                    'Tipo': f'Partición {k+1}',
                    'es_global': False
                })
        for e, acc_val in enumerate(historial_acc, 1):
            acc_data.append({
                'Época': e,
                'Precisión': acc_val,
                'Tipo': 'Global (promediado)',
                'es_global': True
            })
        
        df_acc = pd.DataFrame(acc_data)
        
        # Gráfica de pérdida con capas
        loss_particiones = alt.Chart(df_loss[~df_loss['es_global']]).mark_line(
            opacity=0.3, size=1
        ).encode(
            x='Época:Q',
            y='Loss:Q',
            color=alt.Color('Tipo:N', legend=alt.Legend(labelFontSize=7))
        )
        
        loss_global = alt.Chart(df_loss[df_loss['es_global']]).mark_line(
            color='black', size=2.5
        ).encode(
            x='Época:Q',
            y='Loss:Q'
        )
        
        loss_chart = alt.layer(loss_particiones, loss_global).properties(
            title='Pérdida (Loss) — Algoritmo de Diego',
            width=700,
            height=350
        ).encode(
            x=alt.X('Época:Q', title='Época'),
            y=alt.Y('Loss:Q', title='Cross-Entropy Loss')
        )
        
        # Gráfica de precisión train con capas
        acc_particiones = alt.Chart(df_acc[~df_acc['es_global']]).mark_line(
            opacity=0.3, size=1
        ).encode(
            x='Época:Q',
            y='Precisión:Q',
            color=alt.Color('Tipo:N', legend=alt.Legend(labelFontSize=7))
        )
        
        acc_global = alt.Chart(df_acc[df_acc['es_global']]).mark_line(
            color='black', size=2.5
        ).encode(
            x='Época:Q',
            y='Precisión:Q'
        )
        
        acc_chart = alt.layer(acc_particiones, acc_global).properties(
            title='Precisión Train — Algoritmo de Diego',
            width=700,
            height=350
        ).encode(
            x=alt.X('Época:Q', title='Época'),
            y=alt.Y('Precisión:Q', title='Precisión (%)', scale=alt.Scale(domain=[0, 100]))
        )
        
        # Combinar gráficas
        chart = alt.hconcat(loss_chart, acc_chart)
        
        # Guardar como HTML
        chart.save('entrenamiento_diego.html')
        print("\n  Gráfica guardada en 'entrenamiento_diego.html'")

    except ImportError:
        print("\n  (altair no disponible, se omite la gráfica)")
