# -*- coding: utf-8 -*-

"""Sequential Estimator"""
import hw3_1a

def add_data(data,pre_mean,pre_m2,data_num):
    cur_mean=(pre_mean+(data-pre_mean)/data_num)
    cur_m2=pre_m2+(data-pre_mean)*(data-cur_mean)
    return cur_mean,cur_m2

def main():

    mean=float(input("mean:"))
    var=float(input("variance:"))
    data_num=0
    cur_mean=0
    cur_m2=0
    pre_mean=0
    pre_m2=0
    cur_var=0
    pre_var=0
    error=0.0001
    print("Data point source function: N("+str(mean)+", "+str(var)+")")
    while True:
        data = float(hw3_1a.normal_generating(mean, var))
        data_num=data_num+1
        print('Add data point: '+str(data))
        pre_mean,pre_m2,pre_var=cur_mean,cur_m2,cur_var
        cur_mean,cur_m2=add_data(data,pre_mean,pre_m2,data_num)

        if data_num>1:
            cur_var=cur_m2/(data_num-1)
        else:
            cur_var=0
        print('Mean = '+str(cur_mean)+' Variance = '+str(cur_var))

        if (abs(cur_mean-pre_mean) < error) and (abs(cur_var-pre_var) < error):
            break

if __name__ == '__main__':
    main()
