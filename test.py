
import multiopti as mop

mo = mop.multiopti(1,3,200) #create an instance of class multiopt
mo.ref_indx()
mo.EM()
mo.DBR()
mo.calc()
mo.plot_reslt()