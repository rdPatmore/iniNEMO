import xarray as xr
import matplotlib.pyplot as plt
import config

class oscilations(object):

    def __init__(self, case):
        self.case = case
        file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
        self.preamble = config.data_path() + case + '/' + file_id

    def get_vels(self):
        self.u = xr.open_dataset(self.preamble + 'grid_U.nc',
                                 chunks={'time_counter':10} ).uo
        self.v = xr.open_dataset(self.preamble + 'grid_V.nc',
                                 chunks={'time_counter':10} ).vo
        self.w = xr.open_dataset(self.preamble + 'grid_W.nc',
                                 chunks={'time_counter':10} ).wo

    def get_sea_ice(self):
        self.icemsk = xr.open_dataset(self.preamble + 'icemod.nc',
                                 chunks={'time_counter':10} ).icepres

    def restrict_pos(self, pos):
        self.u = self.u.isel(x=pos[0],y=pos[1],depthu=pos[2]).load()
        self.v = self.v.isel(x=pos[0],y=pos[1],depthv=pos[2]).load()
        self.w = self.w.isel(x=pos[0],y=pos[1],depthw=pos[2]).load()
        self.icemsk = self.icemsk.isel(x=pos[0],y=pos[1]).load()



    def plot_vels_and_si(self):
        pos = [150,150,0]
        self.get_vels()
        self.get_sea_ice()
        self.restrict_pos(pos)

        fig, axs = plt.subplots(2)
        axs[0].plot(self.u.time_counter, self.u, label='U')
        axs[0].plot(self.v.time_counter, self.v, label='V')
        axs[0].plot(self.w.time_counter, self.w, label='W')
        axs[1].plot(self.icemsk.time_counter, self.icemsk, label='ice')

        pos_str = str(pos[0]) + '_' + str(pos[1]) + '_' + str(pos[2])
        plt.savefig('velocity_ocilations_' + pos_str + '.png')

osc = oscilations('EXP10')
osc.plot_vels_and_si()
