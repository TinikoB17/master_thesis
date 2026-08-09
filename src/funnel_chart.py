import plotly.graph_objects as go


stages = [
    "miRNA Expression",
    "Age Data Available",
    "Blood Samples",
    "Sex Data Available",
    "Disease Condition Available",
    "Healthy Samples"
]

counts = [46997,
          14374,
          9335,
          9222,
          7150,
          984
          ]

fg = go.Figure(go.Funnel(y=stages,
                         x=counts,
                         texttemplate="%{x:,}",
                         textinfo="text"))

fg.update_layout(
                template="simple_white",
                font=dict(color="black", size=14),
                plot_bgcolor="white")

fg.write_image('samples_funnel.pdf')
