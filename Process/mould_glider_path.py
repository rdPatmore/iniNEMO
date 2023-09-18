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

        ## remove duplicate index values
        #_, index = np.unique(self.giddy_raw['distance'], return_index=True)
        #group = group.isel(ctd_depth=index)
        
        # get final meso transect
        meso_trans = g_trans.where(g_trans.meso_transect==0, drop=True)
        meso_trans = meso_trans.where(g_trans.transect==34, drop=True)

        # subset straight section of transect
        meso_trans = meso_trans.swap_dims({'ctd_data_point':'distance'})
        subset = meso_trans.sel(distance=slice(1.37e6, 1.49e6))

        # shift start time to that of full deployment
        tdiff = subset[0].ctd_time_dt64 - g_trans[0].ctd_time_dt64
        subset['ctd_time_dt64'] = subset.ctd_time_dt64 - tdiff

        def concat_transect(da0, da1, dim='distance', reverse=True):
            ''' concatented long north-south transect '''
            
            # reverse direction of appended data
            if reverse:
                da1 = da1[::-1]
        
                # assert increasing time and distance
                print (da1.ctd_time_dt64[-1].values)
                print (np.diff(da1.ctd_time_dt64).shape)
                print (da1.ctd_time_dt64[-1].values - np.diff(da1.ctd_time_dt64))
                time = np.concatenate([da1.ctd_time_dt64,
                           da1.ctd_time_dt64[-1] - np.diff(da1.ctd_time_dt64)])
                dist = np.concatenate([da.distance,
                               da1.distance[-1] - np.diff(da1.distance)])
                da1['ctd_time_64'] = time
                da1['distance'] = dist

            # shift dates to end of da0
            tdiff =  da0[-1].ctd_time_dt64 - da0[0].ctd_time_dt64
            da1['ctd_time_dt64'] = da1.ctd_time_dt64 + tdiff
            print (da1.ctd_time_dt64 + tdiff)

            # shift distance to end of da0
            ddiff =  da0[-1].distance - da0[0].distance
            da0['distance'] = da1.distance + ddiff
            
            # append
            da = xr.concat([da0, da1], dim=dim)
    
            return da

        direction_sequence = [False,True] * 8

        da = concat_transect(subset, subset, reverse=True)
        for boolean in direction_sequence:
            da = concat_transect(da, subset, reverse=booleen)

        print (da)

  
    
if __name__ == "__main__":
    m = mould_glider_path()
    m.mould_glider_path_to_straight_line()
    
