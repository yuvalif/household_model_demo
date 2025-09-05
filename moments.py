import numpy as np
import constant_parameters as c
from tabulate import tabulate
import cohorts

underline_index = cohorts.cohort.find("_") 
if underline_index == -1:
    cohort = cohorts.cohort
else:
    cohort = cohorts.cohort[:underline_index]

class ActualMoments:
    def __init__(self):
        self.actual_married_moments_w = np.loadtxt("input/married_w"+cohort+".txt")
        self.actual_married_moments_h = np.loadtxt("input/married_h"+cohort+".txt")
        self.actual_unmarried_moments_w = np.loadtxt("input/unmarried_w"+cohort+".txt")
        self.actual_unmarried_moments_h = np.loadtxt("input/unmarried_h"+cohort+".txt")
        self.actual_marr_divorce_moments = np.loadtxt("input/marr_divorce"+cohort+".txt")
        self.actual_school_moments_w = np.loadtxt("input/school_w"+cohort+".txt")
        self.actual_school_moments_h = np.loadtxt("input/school_h"+cohort+".txt")
        self.actual_assortative_moments = np.loadtxt("input/assortative"+cohort+".txt")
        self.actual_kids_distribution_moments_m = np.loadtxt("input/kids_distribution_m" + cohort + ".txt")
        self.actual_kids_distribution_moments_um = np.loadtxt("input/kids_distribution_um" + cohort + ".txt")

class Moments:
    fertility_moments_single = np.zeros((c.max_period_f, c.kids_size_f))
    fertility_moments_married = np.zeros((c.max_period_f, c.kids_size_f))
    fertility_moments = np.zeros((c.max_period_f, c.kids_size_f))
    school_moments_wife = np.zeros((c.max_school_f+1, c.school_size_f))
    school_moments_husband = np.zeros((c.max_school_f+1, c.school_size_f))
    emp_moments_wife_single = np.zeros((c.max_period_f, 3))  # 0 - unemployed, 1 - part time, 2 - full time
    emp_moments_husband_single = np.zeros((c.max_period_f, 3))  # 0 - unemployed, 1 - part time, 2 - full time
    emp_moments_wife_married = np.zeros((c.max_period_f, 3))  # 0 - unemployed, 1 - part time, 2 - full time
    emp_moments_husband_married = np.zeros((c.max_period_f, 3))  # 0 - unemployed, 1 - part time, 2 - full time
    wage_moments_wife_single = np.zeros(c.max_period_f)
    wage_counter_wife_single = np.zeros(c.max_period_f)
    wage_moments_husband_single = np.zeros(c.max_period_f)
    wage_counter_husband_single = np.zeros(c.max_period_f)
    wage_moments_wife_married = np.zeros(c.max_period_f)
    wage_counter_wife_married = np.zeros(c.max_period_f)
    wage_moments_husband_married = np.zeros(c.max_period_f)
    wage_counter_husband_married = np.zeros(c.max_period_f)
    marriage_moments_w = np.zeros(c.max_period_f)
    divorce_moments_w = np.zeros(c.max_period_f)
    marriage_moments_h = np.zeros(c.max_period_f)
    divorce_moments_h = np.zeros(c.max_period_f)
    assortative_moments = np.zeros((c.school_size_f, c.school_size_f))
    assortative_counter = np.zeros(c.school_size_f)
    welfare_moments_employed = np.zeros(c.max_period_f)
    welfare_counter_employed = np.zeros(c.max_period_f)
    welfare_moments_unemployed = np.zeros(c.max_period_f)
    welfare_counter_unemployed = np.zeros(c.max_period_f)


def decimate(arr, first_row, last_row, avg_size):
    cols = arr.shape[1]
    rows = arr.shape[0]
    if first_row < 0 or first_row >= last_row:
        raise ValueError("decimate: invalid first row ", first_row)
    if last_row >= rows:
        raise ValueError("decimate: invalid last row ", last_row, "size", rows)
    result = np.zeros((int((last_row - first_row + 1)/avg_size), cols))
    row_sum = np.zeros(cols)
    result_i = 0
    for i in range(first_row, last_row+1):
        row_sum += arr[i, :]
        if (i - first_row + 1) % avg_size == 0:
            result[result_i, :] = row_sum / avg_size
            row_sum = np.zeros(cols)
            result_i += 1

    return result


def print_table(display_moments, headline, data, headers):
    if not display_moments:
        return
    print(tabulate([[headline]], tablefmt="simple"))
    print(tabulate(data, headers, floatfmt=".2f", tablefmt="simple_grid"))


def calculate_moments(m, verbose, display_moments):
    # calculate employment moments
    # estimated_married_moments_w = np.zeros((c.max_period_f, 8))
    # age_arr = np.arange(17, 17+c.max_period_f).reshape((1, c.max_period_f))
    # print(age_arr)
    actual = ActualMoments()
    if c.cohort_f == 1960:
        max_period_by_cohort = c.max_1960
    elif c.cohort_f == 1970:
        max_period_by_cohort = c.max_1970
    elif c.cohort_f == 1980:
        max_period_by_cohort = c.max_1980
    elif c.cohort_f == 1990 or c.cohort_f == 2000 or c.cohort_f == 2010:
        max_period_by_cohort = c.max_1990

    age_group_description = [ "25-29", "30-34", "35-39", "40-44", "45-49"]
    age_title = np.array([age_group_description]).T
   
   #    MARRIED WOMEN ####
    age_arr = np.arange(17, 17+max_period_by_cohort)
    temp1 = m.wage_moments_wife_married / np.where(m.wage_counter_wife_married != 0, m.wage_counter_wife_married, np.nan)
    temp2 = (m.emp_moments_wife_married.T / np.where(m.marriage_moments_w != 0, m.marriage_moments_w, np.nan)).T
    estimated_married_moments_w = np.c_[age_arr, temp1[0:max_period_by_cohort], temp2[0:max_period_by_cohort,[1, 2]],
                                        actual.actual_married_moments_w[0:max_period_by_cohort,[5, 7, 6]]]

    temp3 = decimate(estimated_married_moments_w, 8, max_period_by_cohort-1, 5)
    mse_wage_wife_married = np.square(np.subtract(temp3[:,1]/1000,temp3[:,4]/1000)).mean()
    mse_emp_wife_married = np.square(np.subtract(100*temp3[:,[2,3]],100*temp3[:,[5,6]])).mean()
    headers = ["Age", "Fitted: Wage", "part", "full", "Actual: Wage",  "part", "full"]
    print_table(display_moments, "married women moments", temp3, headers)
    
    #    MARRIED MEN ####
    temp1 = m.wage_moments_husband_married / np.where(m.wage_counter_husband_married != 0, m.wage_counter_husband_married, np.nan)
    temp2 = (m.emp_moments_husband_married.T / np.where(m.marriage_moments_h != 0, m.marriage_moments_h, np.nan)).T
    estimated_married_moments_h = np.c_[age_arr, temp1[0:max_period_by_cohort], temp2[0:max_period_by_cohort, [1, 2]],
                                        actual.actual_married_moments_h[0:max_period_by_cohort, [5, 7, 6]]]

    temp3 = decimate(estimated_married_moments_h, 8, max_period_by_cohort-1, 5)
    mse_wage_husband_married = np.square(np.subtract(temp3[:,1]/1000,temp3[:,4]/1000)).mean()
    mse_emp_husband_married = np.square(np.subtract(100*temp3[:,[2,3]],100*temp3[:,[5,6]])).mean()
    print_table(display_moments, "married men moments", temp3, headers)
    
    #    MARRIED COUPLE CHILDREN #########################
    age_arr = np.arange(20, 41)
    temp = (m.fertility_moments_married.T / np.where(m.marriage_moments_w != 0, m.marriage_moments_w, np.nan)).T
    estimated_married_kids_moments = np.c_[age_arr, temp[3:24, :],
                                           actual.actual_kids_distribution_moments_m[0:21, 3:7]]
    temp3 = decimate(estimated_married_kids_moments, 5, 20, 5)

    mse_kids_married = np.square(np.subtract(100*temp3[:, [1, 2, 3, 4]], 100*temp3[:, [5, 6, 7, 8]])).mean()
    headers = ["Age", "Fitted: No-Kids", "1-Kid", "2-Kids", "3+Kids", "Actual: No-Kids", "1-Kid", "2-Kids", "3+Kids" ]
    print_table(display_moments, "married couple children moments", temp3, headers)

    ######## TOTAL FERTILITY ########
    #age_arr = np.arange(20, 41)
    #temp = (m.fertility_moments.T / c.DRAW_F).T
    #temp1 = np.c_[age_arr, temp[3:24, :]]
    #table = tabulate(temp1, headers, floatfmt=".2f", tablefmt="simple")
    #print(" total children moments")
    #print(table)

    ############### SINGLES #####################################################
    age_arr = np.arange(17, 17+max_period_by_cohort)
    #    SINGLE WOMEN ##################################
    temp1 = m.wage_moments_wife_single / np.where(m.wage_counter_wife_single != 0, m.wage_counter_wife_single, np.nan)
    temp2 = (m.emp_moments_wife_single.T / (c.DRAW_F - m.marriage_moments_w)).T
    temp3 = m.welfare_moments_employed / np.where(m.welfare_counter_employed != 0, m.welfare_counter_employed, np.nan)
    temp4 = m.welfare_moments_unemployed / np.where(m.welfare_counter_unemployed != 0, m.welfare_counter_unemployed, np.nan)
    estimated_single_moments_w = np.c_[age_arr, temp1[0:max_period_by_cohort], temp2[0:max_period_by_cohort, [1,2]], temp3[0:max_period_by_cohort], temp4[0:max_period_by_cohort],
                                        actual.actual_unmarried_moments_w[0:max_period_by_cohort, [5, 7, 6, 8, 9]]]
    temp5 = decimate(estimated_single_moments_w, 8, max_period_by_cohort-1, 5)
    mse_wage_wife_single = np.square(np.subtract(temp5[:,1]/1000,temp5[:,6]/1000)).mean()
    mse_emp_wife_single = np.square(np.subtract(100*temp5[:,[2,3]],100*temp5[:,[7,8]])).mean()
    headers = ["Fitted:Age", "Wage", "part", "full", "welfare-emp", "welfare-unemp","Actual:Wage", "part", "full", "welfare-emp", "welfare-unemp"]
    print_table(display_moments, "single women moments", temp5, headers)
    #    SINGLE MEN ##################################
    temp1 = m.wage_moments_husband_single / np.where(m.wage_counter_husband_single != 0, m.wage_counter_husband_single, np.nan)
    temp2 = (m.emp_moments_husband_single.T / (c.DRAW_F - m.marriage_moments_h)).T
    estimated_single_moments_h = np.c_[
        age_arr, temp1[0:max_period_by_cohort], temp2[0:max_period_by_cohort, [1, 2]],
        actual.actual_unmarried_moments_h[0:max_period_by_cohort, [5, 7, 6]]]
    temp5 = decimate(estimated_single_moments_h, 8, max_period_by_cohort-1, 5)
    mse_wage_husband_single = np.square(np.subtract(temp5[:,1]/1000,temp5[:,4]/1000)).mean()
    mse_emp_husband_single = np.square(np.subtract(100*temp5[:,[2,3]],100*temp5[:,[5,6]])).mean()
    headers = ["Fitted:Age", "Wage", "part", "full", "Actual:Wage", "part", "full"]
    print_table(display_moments, "single men moments", temp5, headers)
    #######    SINGLE WOMEN CHILDREN ###############
    age_arr = np.arange(20, 41)
    temp1 = (m.fertility_moments_single.T / (c.DRAW_F - m.marriage_moments_w)).T
    estimated_single_kids_moments = np.c_[age_arr, temp1[3:24, :],
                                      actual.actual_kids_distribution_moments_um[0:21,3:7]]
    temp3 = decimate(estimated_single_kids_moments, 3, 20, 5)
    mse_kids_unmarried = np.square(np.subtract(100*temp3[:,[1,2,3,4]],100*temp3[:,[5,6,7,8]])).mean()
    headers = ["Age", "Fitted:No-Kids", "1-Kid", "2-Kids", "3+Kids", "Actual: No-Kids", "1-Kid", "2-Kids", "3+Kids"]
    print_table(display_moments, "single women children moments", temp3, headers)

    ############ WIFE  MARRIED AND DIVORCE MOMENTS ######################################################################################
    age_arr_1970 = np.arange(17, 17+max_period_by_cohort)
    estimated_marr_divorce_moments = np.c_[age_arr_1970, (m.marriage_moments_w.T[0:max_period_by_cohort] / c.DRAW_F),
                                           (m.divorce_moments_w.T[0:max_period_by_cohort] / c.DRAW_F), actual.actual_marr_divorce_moments[0:max_period_by_cohort, 3:5]]

    temp3 = decimate(estimated_marr_divorce_moments, 8, max_period_by_cohort-1, 5)
    mse_marriage_divorce = np.square(np.subtract(100*temp3[:,[1,2]],100*temp3[:,[3,4]])).mean()

    headers = ["Age", "Fitted: marriage", "divorce", "Actual: married", "divorce"]
    print_table(display_moments, "women marriage/divorce moments", temp3, headers)
    ############ HUSBAND MARRIED AND DIVORCED  #######################################################################################
    #estimated_marr_divorce_moments = np.c_[age_arr_1970, (m.marriage_moments_h.T[0:36] / c.DRAW_F),
    #                                       (m.divorce_moments_h.T[0:36] / c.DRAW_F), actual.actual_marr_divorce_moments[
    #                                                                                 :, 3:5]]

    #temp3 = decimate(estimated_marr_divorce_moments, 8, 32, 5)
    #headers = ["Age", "Fitted: marriage", "divorce", "Actual: married", "divorce"]
    #table = tabulate(temp3, headers, floatfmt=".2f", tablefmt="simple")
    #print(table)
    ###################################################################################################


    school_age_arr = np.arange(18, 18+c.max_school)
    estimated_school_moments_w = np.c_[school_age_arr, (m.school_moments_wife[1:15, :] / c.DRAW_F), actual.actual_school_moments_w[:, 3:8]]
    estimated_school_moments_h = np.c_[school_age_arr, (m.school_moments_husband[1:15, :] / c.DRAW_F), actual.actual_school_moments_h[:, 3:8]]

    mse_school_w = np.square(np.subtract(100*estimated_school_moments_w[c.max_school-2:c.max_school,[1,2]],
                                         100*estimated_school_moments_w[c.max_school-2:c.max_school, [3,4]])).mean()
    mse_school_h = np.square(np.subtract(100*estimated_school_moments_h[c.max_school-2:c.max_school,[1,2]],
                                         100*estimated_school_moments_h[c.max_school-2:c.max_school, [3,4]])).mean()

    headers = ["Age", "Fitted: HSD", "HSG", "SC", "CG", "PC", "Actual: HSD", "HSG", "SC", "CG", "PC"]
    data = estimated_school_moments_w[c.max_school-1:c.max_school, :]
    print_table(display_moments, "women education moments", data, headers)
    data = estimated_school_moments_h[c.max_school-1:c.max_school, :]
    print_table(display_moments, "men education moments", data, headers)
    ###################################################################################################
    estimated_assortative_moments = np.c_[m.assortative_moments / np.where(m.assortative_counter != 0, m.assortative_counter, np.nan), actual.actual_assortative_moments[:,1:6]]
    mse_assortative = np.square(np.subtract(100*estimated_assortative_moments[:,0:5],100*estimated_assortative_moments[:, 5:10])).mean()
    print_table(display_moments, "assortative mating: column:wife, row:husband", estimated_assortative_moments, headers)
    ###################################################################################################
    
    objective_function = mse_wage_wife_married + mse_emp_wife_married + \
        mse_wage_husband_married + mse_emp_husband_married +\
        mse_kids_married + \
        mse_wage_wife_single + mse_emp_wife_single + \
        mse_wage_husband_single + mse_emp_husband_single + \
        mse_kids_unmarried + \
        mse_marriage_divorce + \
        mse_school_w + mse_school_h + \
        mse_assortative

    if verbose:
        print("objective function - married:")
        print("wage wife", mse_wage_wife_married)
        print("emp wife", mse_emp_wife_married)
        print("wage husband", mse_wage_husband_married)
        print("emp husband", mse_emp_husband_married)
        print("kids", mse_kids_married)
        print("objective function - unmarried:")
        print("wage wife", mse_wage_wife_single)
        print("emp wife", mse_emp_wife_single)
        print("wage husband", mse_wage_husband_single)
        print("emp husband", mse_emp_husband_single)
        print("kids", mse_kids_unmarried)
        print("objective function:")
        print("marriage and divorce", mse_marriage_divorce)
        print("school wife", mse_school_w)
        print("school husband", mse_school_h)
        print("assortative", mse_assortative)

    return objective_function
