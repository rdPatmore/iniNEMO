import xarray as xr
import model_object
import glidertools as gt
import Plot.get_transects as trans

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

        # first shape is a square
        length_dist = 4000 # meters
         
        self.giddy_raw['distance'] = xr.DataArray( 
                 gt.utils.distance(self.giddy_raw.lon,
                                   self.giddy_raw.lat).cumsum(),
                                   dims='ctd_data_point')
        self.giddy_raw = self.giddy_raw.set_coords("distance")
        self.giddy_raw = self.giddy_raw.swap_dims({"ctd_data_point":"distance"})
        g_trans = trans.get_transects(self.giddy_raw.distance,
                   concat_dim='distance', 
                   method='from interp_1000',
                  shrink=None, drop_trans=[False,False,False,False],
                  offset=False, rotation=None, cut_meso=False)
        print (g_trans)
        print (fkdj)
        self.giddy_raw = self.giddy_raw.set_coords('distance')
        self.giddy_raw = self.giddy_raw.swap_dims(
                                                 {'ctd_data_point': 'distance'})
        print (self.giddy_raw.dives.isel(distance=slice(1,100)))

        # remove duplicate index values
        _, index = np.unique(self.giddy_raw['distance'], return_index=True)
        self.giddy_raw = self.giddy_raw.isel(distance=index)
  
        # iterate over sides
        for i in int(range(giddy_raw.distance.max()/lenth_dist)):
            side = self.giddy_raw.sel(distance=slice(
                         i * length_dist, (i + 1) * length_dist))
        #    if ns:
        #         ds = 

    
if __name__ == "__main__":
    m = mould_glider_path()
    m.mould_glider_path_to_straight_line()
    
