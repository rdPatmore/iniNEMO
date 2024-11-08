from iniNEMO.Plot.Budgets.plot_momentum import plot_momentum
import numpy as np

def check_chamfer():
    file_id = 'CHAMFER_1d_19930101_19930101_flux_form_best_'
    case = 'CHAMFER_flux_form_best'
    date = '19930101'
    mom = plot_momentum(case, file_id, date, mean=False)

    # alter path from default
    mom.preamble = '/gws/nopw/j04/chamfer/AMM15_C09p2_CHAMFER/MomTrd/' + file_id

    # plot budgets
    #mom.plot_mom_budget_slices(vec='u')
    #mom.plot_mom_budget_slices(vec='v')

    # plot residuals 
    mom.plot_mom_residual_budget(vec='u')
    #mom.plot_mom_residual_budget(vec='v')

def check_gyre_trd_4p2p3():
    file_id = 'GYRE_1d_20100101_20101231_'
    case = 'GYRE_TRD_4p2p3_p1'
    date = '20101231'
    mom = plot_momentum(case, file_id, date=date)

    # alter path from default
    path = '/gws/nopw/j04/jmmp/ryapat/MOM_TRD/GYRE_TRD_4p2p3/p1/'
    mom.preamble = path + file_id

    # plot budgets
    mom.plot_mom_budget_slices(vec='u', bdy=False)
    mom.plot_mom_budget_slices(vec='v', bdy=False)

    # plot residuals 
    mom.plot_mom_residual_budget(vec='u', bdy=False)
    mom.plot_mom_residual_budget(vec='v', bdy=False)

#check_chamfer()
check_gyre_trd_4p2p3()
