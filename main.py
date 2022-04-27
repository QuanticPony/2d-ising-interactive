import matplotlib.pyplot as plt
from interactive_ising2d import interactive_ising2d

if __name__ == '__main__':
    ising2d = interactive_ising2d(
        L_min=4, 
        L_initial=16, 
        L_max=64, 
                     
        beta_min=0, 
        beta_max=0.7, 
        delta_beta=0.01
    )

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)
    
    ising2d.prepare_canvas(fig, 
                       ax_temporal=ax3, 
                       ax_spines=ax1, 
                       ax_beta_values=ax4, 
                       ax_widgets=ax2)
    
    ising2d.prepare_animation(montecarlo_steps_per_frame=1)

    plt.show()
