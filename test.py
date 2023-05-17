import multiopti as mop

mo = mop.multiopti(1,3,200) #create an instance of class multiopt
mo.ref_indx(source = 'theory')

# P pol

#_____________
mo.EM()
mo.DBR(Bragg = 555, mode = 1, air_n = 1, DBR_per_up = 4,DBR_per_bot = 4,
              lr1_n = 2, lr2_n = 1.5, cav_n = 2, lr4_n = 1.5, lr5_n = 2, sub_n = 1.5,
              exc_num = 0, exc_thick = 0)
mo.DBRplot()
mo.calc()
mo.plot_reslt()

#mo.plot_0Deg()

