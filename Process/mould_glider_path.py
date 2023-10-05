import xarray as xr
import model_object
import glidertools as gt
import iniNEMO.Plot.get_transects as trans
import numpy as np

class mould_glider_path(model_object.model):
    """
    creating other paths from giddy's full path
    """

    def __init__(self, case="EXP10"):

        super().__init__(case)
        self.load_obs()

        # transect sequence for concatenation
        self.direction_sequence = [False,True] * 8

    def assert_ascending_coord(self, da, coord):
        ''' reverse coordinate in case of descending order '''

        # get first two indicies and coordinate delta 
        ascending_delta_c = - np.cumsum(np.diff(da[coord]))
        c_0 = da[coord][0].values
        c_1 = da[coord][1].values

        # make ascending coordinate
        ascending = np.concatenate(([c_0],(c_1 + ascending_delta_c)))

        # assign data to xarray da
        da[coord] = xr.DataArray(ascending, dims=['distance'])

    def concat_transect(self, da0, da1, dim='distance', reverse=True):
        ''' concatented long north-south transect '''
        
        # reverse direction of appended data
        if reverse:
            da1 = da1[::-1]

            # assert ascending coordinates
            self.assert_ascending_coord(da1, 'ctd_time_dt64')
            self.assert_ascending_coord(da1, 'distance')
            self.assert_ascending_coord(da1, 'dives')
            self.assert_ascending_coord(da1, 'ctd_data_point')

        # shift dates to end of da0
        tdiff =  da0[-1].ctd_time_dt64 - da1[0].ctd_time_dt64
        da1['ctd_time_dt64'] = da1.ctd_time_dt64 + tdiff

        # shift distance to end of da0
        ddiff =  da0[-1].distance - da1[0].distance
        da1['distance'] = da1.distance + ddiff

        # label transect
        transect = np.full(da1.shape, da0.transect[-1] + 1)
        da1['transect'] = xr.DataArray(transect, dims=da1.dims)
        
        # append
        joined = xr.concat([da0, da1], dim=dim)

        return joined

    def mould_glider_path_to_straight_line(self):
        '''
        take distance in glider path reshape the path along distance
        preserving dives and depth
        '''

        self.giddy_raw['distance'] = xr.DataArray( 
                 gt.utils.distance(self.giddy_raw.lon,
                                   self.giddy_raw.lat).cumsum(),
                                   dims='ctd_data_point')
        self.giddy_raw = self.giddy_raw.set_coords(["distance","dives"])

        # get transects
        g_trans = trans.get_transects(self.giddy_raw.distance,
                   concat_dim='ctd_data_point', 
                   method='from interp_1000',
                   shrink=None, drop_trans=[False,False,False,False],
                   offset=False, rotation=None, cut_meso=False)

        # get final meso transect
        meso_trans = g_trans.where(g_trans.meso_transect==0, drop=True)
        meso_trans = meso_trans.where(g_trans.transect==34, drop=True)

        # subset straight section of transect
        meso_trans = meso_trans.swap_dims({'ctd_data_point':'distance'})
        subset = meso_trans.sel(distance=slice(1.37e6, 1.49e6))

        # shift start time to that of full deployment
        tdiff = subset[0].ctd_time_dt64 - g_trans[0].ctd_time_dt64
        subset['ctd_time_dt64'] = subset.ctd_time_dt64 - tdiff
        
        # set initial transect to 0
        transect0 = xr.DataArray(np.full(subset.shape, 0), dims=subset.dims)
        subset0 = subset.assign_coords(transect=transect0)

        # concatenate repeating transects
        da = self.concat_transect(subset0, subset, reverse=True)
        for boolean in self.direction_sequence:
            da = self.concat_transect(da, subset, reverse=boolean)

        # shift coords to start at 0
        da['ctd_data_point'] = da.ctd_data_point - da.ctd_data_point[0]
        da['dives'] = da.dives - da.dives[0]

        # drop irrelevant coords
        drop_c = ['meso_transect', 'vertex']
        self.deployment = da.drop(drop_c)

    def save_deployment(self):
        ''' save new moulded glider path '''

        print (self.deployment)
        f_path = 'Giddy_2020/artificial_straight_line_transects.nc'
        self.deployment.to_netcdf(self.root + f_path)
    
if __name__ == "__main__":
    m = mould_glider_path()
    m.mould_glider_path_to_straight_line()
    m.save_deployment()
