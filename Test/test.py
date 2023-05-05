import multiopti as mop

mo = mop.multiopti(1,3,200) #create an instance of class multiopt
mo.ref_indx(source = 'theory')
# mo.EM()
# mo.DBR(Bragg = 555, mode = 10, air_n = 1, DBR_per_up = 1,DBR_per_bot = 1,
#               lr1_n = 1, lr2_n = 1, cav_n = 1, lr4_n = 1, lr5_n = 1, sub_n = 1,
#               exc_num = 0, exc_thick = 0)

# mo.calc()
# mo.plot_reslt()
# #mo.plot_0Deg()
# #mo.save_text()
# mo.DBRplot()

mo.EM()
mo.DBR(Bragg = 555, mode = 10, air_n = 1.2, DBR_per_up = 3,DBR_per_bot = 2,
              lr1_n = 1.5, lr2_n = 2.5, cav_n = 3, lr4_n = 3.1, lr5_n = 1.3, sub_n = 1.5,
              exc_num = 0, exc_thick = 0)
mo.DBRplot()
mo.calc()
mo.plot_reslt()


# P pol