import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def Plotter_MapOnMap_Plotly(map_r, map_toplot, title=""):
    """
    Plot a color-coded 2D map (map_toplot) projected onto a 3D surface
    defined by map_r, using Plotly. Intended for Streamlit.
    """

    map_shape = map_r.shape

    # Angular coordinates
    pp, tt = np.meshgrid(
        np.linspace(0, 2*np.pi, map_shape[1]),
        np.linspace(0, np.pi, map_shape[0])
    )

    # Cartesian coordinates
    x = map_r * np.sin(tt) * np.cos(pp)
    y = map_r * np.sin(tt) * np.sin(pp)
    z = map_r * np.cos(tt)

    # Normalize color map
#    cmin = np.min(map_toplot)
#    cmax = np.max(map_toplot)
#    colors = (map_toplot - cmin) / (cmax - cmin + 1e-12)

    # Plotly surface
    fig = go.Figure(
        data=go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=map_toplot,#colors,
            colorscale="RdBu",
#            cmin=0,
#            cmax=1,
#            flatshading=True,
            #cmin=max_color,
            #cmax=min_color,
            colorbar=dict(title="Value")
        )
    )

    # Equal aspect ratio
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="µm",
            yaxis_title="µm",
            zaxis_title="µm",
            aspectmode="data",
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig



def Plotter_FlatShade(map_r, map_toplot, title=""):
    map_shape = map_r.shape

    # θ–φ grid
    phi = np.linspace(0, 2*np.pi, map_shape[1])
    theta = np.linspace(0, np.pi, map_shape[0])
    pp, tt = np.meshgrid(phi, theta)

    # Cartesian coordinates
    x = map_r * np.sin(tt) * np.cos(pp)
    y = map_r * np.sin(tt) * np.sin(pp)
    z = map_r * np.cos(tt)

    # Normalize exactly like matplotlib version
    vmin = np.min(map_toplot)
    vmax = np.max(map_toplot)
    map_norm = (map_toplot - vmin) / (vmax - vmin)

    # Flatten
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    c = map_norm.ravel()

    n_theta, n_phi = map_shape

    # Build triangle indices
    I, J, K = [], [], []
    for i in range(n_theta - 1):
        for j in range(n_phi - 1):
            p0 = i*n_phi + j
            p1 = p0 + 1
            p2 = p0 + n_phi
            p3 = p2 + 1

            I += [p0, p0]
            J += [p1, p2]
            K += [p2, p3]

    fig = go.Figure(
        go.Mesh3d(
            x=x, y=y, z=z,
            i=I, j=J, k=K,
            intensity=c,
            colorscale="RdBu",
            cmin=0, cmax=1,
            flatshading=True,
            showscale=True
        )
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="µm",
            yaxis_title="µm",
            zaxis_title="µm",
            aspectmode="data"
        ),
        width=600,
        height=600
    )

    return fig
