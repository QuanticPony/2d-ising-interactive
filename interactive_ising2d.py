import math

import matplotlib
import matplotlib.animation as Animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable


class interactive_ising2d:
    def __init__(self, L_min, L_initial, L_max, beta_min, beta_max, delta_beta):
        """Inits the class"""
        self.L_min = L_min
        self.L_max = L_max
        self.L = L_initial
        self.L_ref = L_initial
        self.beta = None
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.delta_beta = delta_beta

        self._doing_histeresis = False
        self._sentido = True

        self.beta: float
        self.beta_values = np.arange(
            self.beta_min, self.beta_max, self.delta_beta)

        self.create_spins()

        self.probability = []

        self.reset_variables()

    def create_spins(self):
        """Creates a random generated spins vector"""
        # self.spins = np.ones((self.L, self.L), dtype='int')
        self.spins = (np.random.random(
            (self.L_ref, self.L_ref))*2).astype('int')*2-1

    def update_probability(self, beta):
        """Updates the probabilities array for the posible energy changes when a single spin changes."""
        self.probability = [np.exp(-beta * (4*i-8))
                            for i in range(5)] if beta != 0 else [0.5 for i in range(5)]

    def get_probability_index(self, dif_energy):
        """Given the energy diference between a spin flip gives the index needed to obtain the probability of that change."""
        return int((dif_energy+8)/4)

    def metropoli(self):
        """Evolves `simulation.spins` to it's next state using metropoli algorithm."""
        for y in range(self.L):
            for x in range(self.L):
                dif_energy = 2 * self.spins[y, x] * (self.spins[y, (x+1) % self.L] + self.spins[(
                    y-1) % self.L, x] + self.spins[y, (x-1) % self.L] + self.spins[(y+1) % self.L, x])

                C = self.probability[self.get_probability_index(dif_energy)]
                if np.random.random() < C:
                    self.spins[y, x] *= -1

    def energy(self):
        """Returns the energy"""
        return sum(-self.spins[y, x] * (self.spins[y, (x+1) % self.L] + self.spins[(y+1) % self.L, x]) for x in range(self.L) for y in range(self.L)) / (2 * self.L**2)

    def magnetization(self):
        """Returns the magnetization"""
        return np.sum(self.spins)/(self.L**2)

    def prepare_canvas(self, fig, ax_temporal, ax_spins, ax_beta_values, ax_widgets):
        """Adds `fig`, `ax_temporal` and `ax_spins` to the `simulation` object as atributes."""
        self.ax_temporal = ax_temporal
        self.ax_spins = ax_spins
        self.ax_beta_values = ax_beta_values
        self.ax_widgets = ax_widgets
        self.fig = fig
        self.line = False
        self.paused = False

        self.cnorm = matplotlib.colors.Normalize(vmin=-1, vmax=1)

        divider = make_axes_locatable(self.ax_widgets)
        self.ax_pause_button = divider.append_axes("top", size="50%", pad=0.1)
        self.histeresis_button_ax = divider.append_axes(
            "top", size="50%", pad=0.1)
        self.beta_slider_ax = divider.append_axes("top", size="50%", pad=0.1)
        self.L_slider_ax = divider.append_axes("top", size="50%", pad=0.1)

        self.beta_slider = Slider(
            ax=self.beta_slider_ax,
            label=r'$\beta$',
            valmin=self.beta_min,
            valmax=self.beta_max,
            valstep=self.delta_beta,
            valinit=(self.beta_max-self.beta_min)/2,
        )
        self.L_slider = Slider(
            ax=self.L_slider_ax,
            label=r'$L$',
            valmin=self.L_min,
            valmax=self.L_max,
            valstep=1,
            valinit=self.L,
        )

        self.L_slider.on_changed(self.change_L)
        self.beta_slider.on_changed(self.change_beta)

        self.histeresis_button = Button(
            self.histeresis_button_ax, 'Histeresis', hovercolor='0.975')
        self.histeresis_button.on_clicked(self.histeresis)

        self.reset_button = Button(
            self.ax_widgets, 'Reset', color=(0.75, 0, 0))
        self.reset_button.on_clicked(self.reset_variables)

        self.pause_button = Button(
            self.ax_pause_button, 'Pause/Continue', color=(0, 0.75, 0.75))
        self.pause_button.on_clicked(self.toggle_pause)

        self.axis_format()

    def change_beta(self, val):
        """Changes the beta used in the simulation"""
        self.beta = val

    def change_L(self, val):
        """Changes the beta used in the simulation"""
        self.L_ref = int(val)
        self.create_spins()

    def axis_format(self):
        """Gives axis title and label names"""
        self.ax_spins.set_title('spins')
        self.ax_spins.set_ylabel('')
        self.ax_spins.set_xlabel('')
        self.ax_temporal.set_title(r'Thermalization')
        self.ax_temporal.set_xlabel(r'Montecarlo iterations')
        self.ax_temporal.set_ylabel(r'Value')
        self.ax_temporal.set_ylim(-1, 1)
        self.ax_beta_values.set_title(r'Values by $\beta$')
        self.ax_beta_values.set_xlabel(r'$\beta$')
        self.ax_beta_values.set_ylabel(r'Value')
        self.ax_beta_values.set_xlim(self.beta_min, self.beta_max)
        self.ax_beta_values.set_ylim(-1, 1)

    def prepare_animation(self, montecarlo_steps_per_frame=1, interval=50):
        """Creates `matplotlib.animation.Animation` and asigns the simulation montecarlo steps per frame."""
        self.montecarlo_steps_per_frame = montecarlo_steps_per_frame
        self.animation = Animation.FuncAnimation(
            self.fig, self.update_plots, interval=interval)

    def plot_temporal(self):
        """Plots the historial energy and magnetization"""
        self.ax_temporal.plot(self.iteration_counter,
                              self.energy_historial, '.', color='blue')
        self.ax_temporal.plot(self.iteration_counter,
                              self.magnetization_historial, '.', color='purple')
        self.ax_temporal.errorbar(self.iteration_counter, self.energy_historial, yerr=np.sqrt(
            self.energy_variance), fmt='none', color='blue', label='Energy')
        self.ax_temporal.errorbar(self.iteration_counter, self.magnetization_historial, yerr=np.sqrt(
            self.magnetization_variance), fmt='none', color='purple', label='Magnetization')
        self.ax_temporal.legend()

    def histeresis(self, *args):
        """Restarts variables and starts histeresis"""
        self.reset_variables()
        self.beta = self.beta_min
        self._doing_histeresis = True
        self._sentido = True

    def plot_beta_values(self):
        """Plots stored values in function of beta"""

        if self._doing_histeresis:
            if not self.line:
                self._X, self._Y = [], [[], []]
                self.line = True

            self._X.append(self.beta)
            for i in range(2):
                self._Y[i].append(self.beta_values_historial[int(
                    self.beta/self.delta_beta)][i][-1])
            self.ax_beta_values.plot(self._X, self._Y[0], color='blue')
            self.ax_beta_values.plot(self._X, self._Y[1], color='purple')
            return

        self.line = False
        for i, b in enumerate(self.beta_values):
            if length := len(self.beta_values_historial[i][0]):

                formater = '-' if self._doing_histeresis else '.'
                self.ax_beta_values.plot(
                    [b for _ in range(length)], self.beta_values_historial[i][0], formater, color='blue')
                self.ax_beta_values.plot(
                    [b for _ in range(length)], self.beta_values_historial[i][1], formater, color='purple')

                if not self._doing_histeresis:
                    self.ax_temporal.errorbar([b for _ in range(length)], self.beta_values_historial[i][0], yerr=np.sqrt(
                        self.beta_values_variances[i][0]), fmt='none', color='blue')
                    self.ax_temporal.errorbar([b for _ in range(length)], self.beta_values_historial[i][1], yerr=np.sqrt(
                        self.beta_values_variances[i][1]), fmt='none', color='purple')

    def reset_variables(self, *args):
        """Empties historial arrays of magnetization and energy"""
        self.iteration_counter = []
        self.energy_historial = []
        self.energy_variance = []
        self.magnetization_historial = []
        self.magnetization_variance = []

        self.beta_values_historial = [[list(), list()]
                                      for i in self.beta_values]
        self.beta_values_variances = [[list(), list()]
                                      for i in self.beta_values]

        self._X, self._Y = [], [[], []]

    def plot_spins(self):
        """Plots `self.spins`"""
        self.ax_spins.imshow(self.spins, cmap='inferno', norm=self.cnorm)

    def toggle_pause(self, *args, **kwargs):
        """Pause or resume the simulation"""
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused

    def update_plots(self, *args, iterations=None, **kargs):
        """Function in charge of updating the plos."""
        self.L = self.L_ref

        if iterations is None:
            iterations = self.montecarlo_steps_per_frame

        if self.beta is None:
            self.beta = (self.beta_max-self.beta_min)/2

        self.update_probability(self.beta)
        for _ in range(iterations):
            self.metropoli()

            e, m = self.energy(), self.magnetization()
            beta_index = math.floor((self.beta-self.beta_min)/self.delta_beta)
            self.energy_historial.append(e)
            self.magnetization_historial.append(m)

            self.beta_values_historial[beta_index][0].append(e)
            self.beta_values_historial[beta_index][1].append(m)

            if len(self.iteration_counter) > 2:
                ev = np.var(
                    self.energy_historial[max(-20, -len(self.energy_historial)):-1])

                mv = np.var(
                    self.magnetization_historial[max(-20, -len(self.magnetization_historial)):-1])
            else:
                ev = 0
                mv = 0

            self.energy_variance.append(ev)
            self.magnetization_variance.append(mv)
            self.beta_values_variances[beta_index][0].append(ev)
            self.beta_values_variances[beta_index][1].append(mv)

            try:
                self.iteration_counter.append(self.iteration_counter[-1]+1)
            except IndexError as e:
                self.iteration_counter.append(0)

        self.ax_temporal.clear()
        self.plot_temporal()

        self.ax_spins.clear()
        self.plot_spins()

        self.ax_beta_values.clear()
        self.plot_beta_values()

        self.axis_format()

        if self._doing_histeresis:
            if self._sentido:
                self.beta += self.delta_beta
                if self.beta > self.beta_max:
                    self.beta -= self.delta_beta
                    self._sentido = False

            else:
                self.beta -= self.delta_beta
                if self.beta < self.beta_min:
                    self.beta += self.delta_beta
                    self._sentido = True
