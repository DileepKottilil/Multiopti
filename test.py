
import multiopti as mop

mo = mop.multiopti(1,3,200) #create an instance of class multiopt
mo.ref_indx(source = 'theory')
mo.EM()
mo.DBR()
mo.DBRplot()
mo.calc()
mo.plot_reslt()
# mo.plot_DBR()

# P pol