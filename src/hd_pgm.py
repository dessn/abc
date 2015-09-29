#!/usr/bin/env python

from matplotlib import rc
from daft import PGM, Node, Plate
rc("font", family="serif", size=12)
rc("text", usetex=True)


pgm = PGM([10.5, 7.5], origin=[0., 0.2], observed_style='inner')

pgm.add_node(Node('dispersion',r"\center{$\sigma_{Ia}$ \newline $\sigma_{non-Ia}$}", 1,7,scale=1.2,aspect=1.8))
pgm.add_node(Node('rate',r"$\theta_r$",2,7))
pgm.add_node(Node('theta_T',r"\center{$\alpha_{Ia}$, $\alpha_{non-Ia}$ \newline $\theta_{Ia}$, $\theta_{non-Ia}$}", 5,7,scale=1.4,aspect=1.8))
pgm.add_node(Node('mu',r"$\Omega_M$, $w$", 6,7, scale=1.4))
pgm.add_node(Node('Transmission',r"$Z$", 7, 7))

pgm.add_node(Node('G_i',r"$g_i$", 3,6))
pgm.add_node(Node('G_Ni',r"$g_{Ni}$", 4,1))

pgm.add_node(Node('Gals',r"$\theta_G$", 10,6, observed=True))


pgm.add_node(Node('theta_Ti',r"\center{$\theta_{Ia,i}$ \newline $\theta_{non-Ia,i}$}", 1,5,scale=1.5,aspect=1.2))

pgm.add_node(Node('Type',r"$T_i$", 2, 5))
pgm.add_node(Node('z',r"$z_i$", 3, 3, fixed=True,offset=(-10,-5)))

pgm.add_node(Node('ST',r"${T}_{Si}$", 2, 1, observed=True))
pgm.add_node(Node('Sz',r"${z}_{Si}$", 3, 1, observed=True))
pgm.add_node(Node('Stheta',r"${\theta}_{Si}$", 1, 1,  observed=True))


pgm.add_node(Node('HD',r"$\mu_i$", 5,3,fixed=True,offset=(10,-5)))
pgm.add_node(Node('Luminosity',r"$L_i(t,\lambda)$", 5, 4, fixed=True, scale=1.2, offset=(20,-22)))
pgm.add_node(Node('Flux_g',r"$f_{gi}(\lambda)$", 5, 1,fixed=True,offset=(0,-25)))
pgm.add_node(Node('Flux',r"$f_i(t,\lambda)$", 5, 2, scale=1.2,fixed=True,offset=(20,-25)))

pgm.add_node(Node('Counts_g',r"$\overline{\mathit{ADU}}_{gi}$", 8, 1,scale=1.2,fixed=True,offset=(10,-25)))
pgm.add_node(Node('Counts',r"$\overline{\mathit{ADU}}_i$", 8, 2,scale=1.2,fixed=True,offset=(15,0)))
pgm.add_node(Node('^Type',r"$\tau_i$", 9, 1, fixed=True,offset=(10,-10)))
pgm.add_node(Node('^Counts',r"${\mathit{ADU}_i}$", 9, 2, observed=True,scale=1.2,aspect=1.2))




pgm.add_edge("G_i","Gals")

pgm.add_edge("G_Ni","Gals")

pgm.add_edge("rate","Type")

pgm.add_edge("Type", "theta_Ti")


pgm.add_edge("mu","HD")
pgm.add_edge("^Counts","^Type")
pgm.add_edge("dispersion", "theta_Ti")
#pgm.add_edge("Host","Luminosity")


pgm.add_edge("z","HD")
pgm.add_edge("z","Sz")

# pgm.add_edge("Coords","G_i")
#pgm.add_edge("G_i","Host")

pgm.add_edge("G_i","z")
pgm.add_edge("HD","Flux")
pgm.add_edge("z","Flux")
pgm.add_edge("G_Ni","Flux_g")
pgm.add_edge("HD","Flux_g")
pgm.add_edge("z","Flux_g")
#pgm.add_edge("HD","^Host")


pgm.add_edge("G_i","Type")
pgm.add_edge("G_i","theta_Ti")

pgm.add_edge("Type","Luminosity")
pgm.add_edge("Luminosity","Flux")


pgm.add_edge("theta_T","Luminosity")
#pgm.add_edge("Galaxies","G")

pgm.add_edge("theta_Ti","Luminosity")


pgm.add_edge("Flux","Counts")
pgm.add_edge("Flux_g","Counts_g")

pgm.add_edge("Counts","^Counts")
pgm.add_edge("Counts_g","^Counts")


pgm.add_edge("Transmission","Counts")
pgm.add_edge("Transmission","Counts_g")
#pgm.add_edge("Transmission","Zeropoints")

pgm.add_edge("Type","ST")
pgm.add_edge("theta_Ti","Stheta")

pgm.add_edge("Counts_g","^Counts")


#pgm.add_edge("Detected","^Counts")




# Big Plate: Galaxy
pgm.add_plate(Plate([0.5, 0.5, 9, 6.],
                    label=r"SNe $i = 1, \cdots, N_{SN}$",
                    shift=-0.2,label_offset=[180,2]))


# Render and save.
pgm.render()
# pgm.figure.text(0.2,0.98,r'\underline{UNIVERSE}',size='large')
# pgm.figure.text(0.45,0.98,r'\underline{OBSERVATORY}',size='large')
# pgm.figure.text(0.72,0.98,r'\underline{DATA}',size='large')

pgm.figure.savefig("../results/hdpgm.pdf")
