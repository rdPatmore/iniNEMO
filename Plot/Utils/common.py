import numpy as np

def time_mean_to_orca(orca, sochic):
    # state clear names
    bins0       = orca.time_counter_bounds.isel(axis_nbounds=0)
    bins1       = orca.time_counter_bounds.isel(axis_nbounds=1)
    bins  = np.union1d(bins0,bins1)
    bin_labels = orca.time_counter.values

    print (orca.time_counter)
    print (bins0)
    print (bins1)

    # mean over orca time periods
    sochic_bin = sochic.groupby_bins('time_counter', bins, labels=bin_labels)
    sochic_mean = sochic_bin.mean()
    sochic_mean = sochic_mean.rename_dims({'time_counter_bins': 'time_counter'})
    return sochic_mean

def monthly_mean(case):
    return case.groupby('time_counter.month').mean()
