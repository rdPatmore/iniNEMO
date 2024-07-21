from iniNEMO.Plot.Budgets.plot_momentum import plot_momentum

def check_chamfer():
    file_id = 'CHAMFER_1d_19930101_19930107_flux_form_dynspg_fix_'
    case = 'CHAMFER_flux_form_dynspg_fix'
    mom = plot_momentum(case, file_id)

    # alter path from default
    mom.preamble = '/gws/nopw/j04/chamfer/AMM15_C09p2_CHAMFER/MomTrd/' + file_id

    # plot budgets
    mom.plot_mom_budget_slices(vec='u')
    #mom.plot_mom_budget_slices(vec='v')

    # plot residules 
    mom.plot_mom_residule_budget(vec='u')
    #mom.plot_mom_residule_budget(vec='v')

def check_gyre_trd_4p2p3():
    file_id = 'GYRE_1d_20100101_20101231_'
    case = 'GYRE_TRD_4p2p3'
    mom = plot_momentum(case, file_id)

    # alter path from default
    mom.preamble = '/gws/nopw/j04/jmmp/ryapat/MOM_TRD/GYRE_TRD_4p2p3/' + file_id

    # plot budgets
    mom.plot_mom_budget_slices(vec='u')
    mom.plot_mom_budget_slices(vec='v')

    # plot residules 
    mom.plot_mom_residule_budget(vec='u')
    mom.plot_mom_residule_budget(vec='v')

check_chamfer()
#check_gyre_trd_4p2p3()
