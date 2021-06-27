
import multiopti as mop

mo = mop.multiopti(1,3,200) #create an instance of class multiopt
mo.ref_indx(source = 'theory')
mo.EM()
mo.DBR(Bragg = 555, mode = 1, air_n = 1, DBR_per_up = 6,DBR_per_bot = 2,
              lr1_n = 1.5, lr2_n = 2.5, cav_n = 3, lr4_n = 2.5, lr5_n = 1.5, sub_n = 1.5,
              exc_num = 3, exc_thick = 10)
mo.DBRplot()
mo.calc()
mo.plot_reslt()
# mo.plot_DBR()

# P pol