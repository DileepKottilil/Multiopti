#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 18:40:21    2021

@author: kottilil dileep and team
"""
import math
import time
import sys
import numpy as np
# import pandas as pd
from numpy.core.defchararray import endswith
from numpy.matrixlib.defmatrix import asmatrix
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm

class multiopti:
    def __init__(self, start,stop,num):
      self.start = start
      self.stop = stop
      self.num = num
      np.set_printoptions(threshold=11)
      self.ad_freespace = 2.6544*1E-3 # optical admittance of free space

    
    def c_map(self,val):
      
      tmax = 230
      tmin = 15
      deno = (self.rmax-self.rmin)
      
      if deno == 0:


        mm = (val-self.rmin)
      else:
        mm = (val-self.rmin)/(self.rmax-self.rmin)*(tmax-tmin)+tmin

      r = round(math.sin(0.024 * mm + 0) * 127 + 128)
      g = round(math.sin(0.024 * mm + 2) * 127 + 128)
      b = round(math.sin(0.024 * mm + 4) * 127 + 128)
      return self.rgb2hex(r,g,b)

    def rgb2hex(self,r,g,b):
      r = int(r)
      g = int(g)
      b = int(b)
      return "#{:02x}{:02x}{:02x}".format(r,g,b)

    """ def DBRplot(self,ax = None):
      if ax is None:
            fig, ax = plt.subplots()
      else:
            fig = ax.figure

      y1 = 0
      lst = [self.air_n,self.lr1_n,self.lr2_n,self.lr4_n,self.lr5_n,self.cav_n,self.sub_n]
      self.rmin = min(lst)
      self.rmax = max(lst)

    
      ax.annotate(str(self.air_n), (25, self.thick_layer1/2), color='b', weight='bold', 
                 fontsize=6, ha='center', va='center')
      for i in range(int(self.DBR_per_up)): #changed
        rectangle1 = plt.Rectangle((0,-y1), 50, -self.thick_layer1, fc=self.c_map(self.lr1_n),ec="black")
        rectangle2 = plt.Rectangle((0,-y1-self.thick_layer1), 50, -self.thick_layer2, fc=self.c_map(self.lr2_n),ec="black")


        ax.add_patch(rectangle1)
        ax.add_patch(rectangle2)

        ax.annotate(str(self.lr1_n), (55, -y1-self.thick_layer1/2), color='b', weight='bold', 
                 fontsize=6, ha='center', va='center')
        ax.annotate(str(self.lr2_n), (55, -y1-self.thick_layer1-self.thick_layer2/2), color='b', weight='bold', 
                 fontsize=6, ha='center', va='center')
        y1 = y1+self.thick_layer1+self.thick_layer2

        if self.DBR_per_up - int(self.DBR_per_up) > 0:
          
           
          rectangle1 = plt.Rectangle((0,-y1), 50, -self.thick_layer1, fc=self.c_map(self.lr1_n),ec="black")

          ax.add_patch(rectangle1)
          
          ax.annotate(str(self.lr1_n), (55, -y1-self.thick_layer1/2), color='b', weight='bold', 
                 fontsize=6, ha='center', va='center')
          y1 = y1+self.thick_layer1

      if self.exc_num != 0 and self.exc_thick != 0:

        for i in range(math.floor(self.cav_layers/2)):
          rectangle1 = plt.Rectangle((0,-y1), 50, -self.cav_layer_thick, fc=self.c_map(self.cav_n),ec="black")
          rectangle2 = plt.Rectangle((0,-y1-self.cav_layer_thick), 50, -self.exc_thick, fc='green',ec="black")

          ax.add_patch(rectangle1)
          ax.add_patch(rectangle2)
          
          ax.annotate(str(self.cav_n), (55, -y1-self.cav_layer_thick/2), color='b', weight='bold', 
                  fontsize=6, ha='center', va='center')
          ax.annotate("Matter", (55, -y1-self.cav_layer_thick-self.exc_thick/2), color='b', weight='bold', 
                  fontsize=6, ha='center', va='center')
          y1 = y1+self.cav_layer_thick+self.exc_thick
      
      elif self.exc_num == 0 or self.exc_thick == 0:
        for i in range(math.floor(self.cav_layers/2)):
          rectangle1 = plt.Rectangle((0,-y1), 50, -self.cav_layer_thick, fc=self.c_map(self.cav_n),ec=self.c_map(self.cav_n))
          # rectangle2 = plt.Rectangle((0,-y1-self.cav_layer_thick), 50, -self.exc_thick, fc='green',ec="black")

          ax.add_patch(rectangle1)
          # plt.gca().add_patch(rectangle2)
          
          ax.annotate(str(self.cav_n), (55, -y1-self.cav_layer_thick/2), color='b', weight='bold', 
                  fontsize=6, ha='center', va='center')
          # ax.annotate("Matter", (55, -y1-self.cav_layer_thick-self.exc_thick/2), color='b', weight='bold', 
                  # fontsize=6, ha='center', va='center')
          y1 = y1+self.cav_layer_thick

        
      rectangle2 = plt.Rectangle((0,-y1), 50, -self.cav_layer_thick, fc=self.c_map(self.cav_n),ec=self.c_map(self.cav_n))
      ax.add_patch(rectangle2)
      
      ax.annotate(str(self.cav_n), (55, -y1-self.cav_layer_thick/2), color='b', weight='bold', 
                 fontsize=6, ha='center', va='center')
      
      y1 = y1+self.cav_layer_thick

      
      for i in range(int(self.DBR_per_bot)):
        rectangle1 = plt.Rectangle((0,-y1), 50, -self.thick_layer4, fc=self.c_map(self.lr4_n),ec="black")
        rectangle2 = plt.Rectangle((0,-y1-self.thick_layer4), 50, -self.thick_layer5, fc=self.c_map(self.lr5_n),ec="black")

        ax.add_patch(rectangle1)
        ax.add_patch(rectangle2)

        ax.annotate(str(self.lr4_n), (55, -y1-self.thick_layer4/2), color='b', weight='bold', 
                 fontsize=6, ha='center', va='center')
        ax.annotate(str(self.lr5_n), (55, -y1-self.thick_layer4-self.thick_layer5/2), color='b', weight='bold', 
                 fontsize=6, ha='center', va='center')

        y1 = y1+self.thick_layer4+self.thick_layer5

        if self.DBR_per_bot - int(self.DBR_per_bot) > 0:
        
          
          rectangle1 = plt.Rectangle((0,-y1), 50, -self.thick_layer4, fc=self.c_map(self.lr4_n),ec="black")

          ax.add_patch(rectangle1)
          
          ax.annotate(str(self.lr4_n), (55, -y1-self.thick_layer4/2), color='b', weight='bold', 
                  fontsize=6, ha='center', va='center')
          y1 = y1+self.thick_layer4

        # plt.axis('scaled')
      ax.annotate(str(self.sub_n), (25, -y1-self.thick_layer5/2), color='b', weight='bold', 
                 fontsize=6, ha='center', va='center')
      ax.autoscale()
      ax.set_xlim(0,70)

      return fig, ax """
    
    def DBRplot(self, ax = None):
      if ax is None:
          fig, ax = plt.subplots()
      else:
          fig = ax.figure

      y1 = 0
      lst = [self.air_n,self.lr1_n,self.lr2_n,self.lr4_n,self.lr5_n,self.cav_n,self.sub_n,self.finite_sub_indx[0][0]]
      self.rmin = min(lst)
      self.rmax = max(lst)

      ax.annotate(str(self.air_n), (25, self.thick_layer1/2), color='b', weight='bold', 
                  fontsize=6, ha='center', va='center')
      
      top_full_pairs = int(self.DBR_per_up_for_schematic)
      top_extra_layer = self.DBR_per_up_for_schematic - top_full_pairs

      for i in range(top_full_pairs):
          for layer_thickness, layer_ri in [(self.thick_layer1, self.lr1_n), (self.thick_layer2, self.lr2_n)]:
              rectangle = plt.Rectangle((0,-y1), 50, -layer_thickness, fc=self.c_map(layer_ri),ec="black")
              ax.add_patch(rectangle)
              ax.annotate(str(layer_ri), (55, -y1-layer_thickness/2), color='b', weight='bold', 
                          fontsize=6, ha='center', va='center')
              y1 += layer_thickness

      if top_extra_layer > 0:
          rectangle = plt.Rectangle((0,-y1), 50, -self.thick_layer1, fc=self.c_map(self.lr1_n),ec="black")
          ax.add_patch(rectangle)
          ax.annotate(str(self.lr1_n), (55, -y1-self.thick_layer1/2), color='b', weight='bold', 
                      fontsize=6, ha='center', va='center')
          y1 += self.thick_layer1

      # Similar logic for cavity and exciton layers...
      if self.exc_num != 0 and self.exc_thick != 0:

        for i in range(math.floor(self.cav_layers/2)):
          rectangle1 = plt.Rectangle((0,-y1), 50, -self.cav_layer_thick, fc=self.c_map(self.cav_n),ec="black")
          rectangle2 = plt.Rectangle((0,-y1-self.cav_layer_thick), 50, -self.exc_thick, fc='green',ec="black")

          ax.add_patch(rectangle1)
          ax.add_patch(rectangle2)
          
          ax.annotate(str(self.cav_n), (55, -y1-self.cav_layer_thick/2), color='b', weight='bold', 
                  fontsize=6, ha='center', va='center')
          ax.annotate("Matter", (55, -y1-self.cav_layer_thick-self.exc_thick/2), color='b', weight='bold', 
                  fontsize=6, ha='center', va='center')
          y1 = y1+self.cav_layer_thick+self.exc_thick
      
      elif self.exc_num == 0 or self.exc_thick == 0:
        for i in range(math.floor(self.cav_layers/2)):
          rectangle1 = plt.Rectangle((0,-y1), 50, -self.cav_layer_thick, fc=self.c_map(self.cav_n),ec=self.c_map(self.cav_n))
          # rectangle2 = plt.Rectangle((0,-y1-self.cav_layer_thick), 50, -self.exc_thick, fc='green',ec="black")

          ax.add_patch(rectangle1)
          # plt.gca().add_patch(rectangle2)
          
          ax.annotate(str(self.cav_n), (55, -y1-self.cav_layer_thick/2), color='b', weight='bold', 
                  fontsize=6, ha='center', va='center')
          # ax.annotate("Matter", (55, -y1-self.cav_layer_thick-self.exc_thick/2), color='b', weight='bold', 
                  # fontsize=6, ha='center', va='center')
          y1 = y1+self.cav_layer_thick

        
      rectangle2 = plt.Rectangle((0,-y1), 50, -self.cav_layer_thick, fc=self.c_map(self.cav_n),ec=self.c_map(self.cav_n))
      ax.add_patch(rectangle2)
      
      ax.annotate(str(self.cav_n), (55, -y1-self.cav_layer_thick/2), color='b', weight='bold', 
                 fontsize=6, ha='center', va='center')
      
      y1 = y1+self.cav_layer_thick
###########
      
      #delete -1 and bottom_extra_layer =  0.5 later
      bottom_full_pairs = int(self.DBR_per_bot_for_schematic)-1
      #bottom_extra_layer = self.DBR_per_bot_for_schematic - bottom_full_pairs
      bottom_extra_layer = -0.5

      for i in range(bottom_full_pairs):
          for layer_thickness, layer_ri in [(self.thick_layer4, self.lr4_n), (self.thick_layer5, self.lr5_n)]:
              rectangle = plt.Rectangle((0,-y1), 50, -layer_thickness, fc=self.c_map(layer_ri),ec="black")
              ax.add_patch(rectangle)
              ax.annotate(str(layer_ri), (55, -y1-layer_thickness/2), color='b', weight='bold', 
                          fontsize=6, ha='center', va='center')
              y1 += layer_thickness

      if bottom_extra_layer > 0:
          rectangle = plt.Rectangle((0,-y1), 50, -self.thick_layer4, fc=self.c_map(self.lr4_n),ec="black")
          ax.add_patch(rectangle)
          ax.annotate(str(self.lr4_n), (55, -y1-self.thick_layer4/2), color='b', weight='bold', 
                      fontsize=6, ha='center', va='center')
          y1 += self.thick_layer4
          
    # The rest of your code remains unchanged...

##########     

      # New conditional layer
      if self.finite_sub_thick[0][0]!=0:
          rectangle = plt.Rectangle((0,-y1), 50, -self.finite_sub_thick[0][0], fc=self.c_map(self.finite_sub_indx[0][0]),ec="black")
          ax.add_patch(rectangle)
          ax.annotate(str(self.finite_sub_indx[0][0]), (55, -y1-self.finite_sub_thick[0][0]/2), color='b', weight='bold', 
                      fontsize=6, ha='center', va='center')
          y1 += self.finite_sub_thick[0][0]
      # Final layers and annotation...
      #substrate plot
      """ rectangle = plt.Rectangle((0,-y1), 50, -0.2E-6, fc=self.c_map(self.sub_n),ec="black")
      ax.add_patch(rectangle) """
      ax.annotate(str(self.sub_n), (25, -y1-self.thick_layer5/2), color='b', weight='bold', 
                 fontsize=6, ha='center', va='center')
      time.sleep(0.1)
      ax.autoscale()
      ax.set_xlim(0,70)
      #ax.set_ylim(-4E-6,0)

      return fig, ax
      


    def Reverse(self,lst):
      new_lst = lst[::-1]
      return new_lst

    def file_path(self,filepath, source1 = 'd'):
      # self.filepath = filepath
      self.file = open(filepath,"r")
      print(filepath)
      return
    
    def ref_indx(self,source = 'theory', fileName = None,e0 = 3,f = 10,gam = 0.02,exc = 2.238,draw = 0):
      self.source = source
      if self.source =='theory':

        self.omega = np.linspace(self.start,self.stop,self.num) #1,3
        self.e0 = e0
        self.f = f
        self.gam = gam
        self.exc = exc
    
        self.eps = self.e0+(self.f/((self.exc**2-self.omega**2)-1j*self.gam*self.omega))
        
        self.refidx = np.sqrt(self.eps)
        self.n1 = np.real(self.refidx)
        self.k1 = np.imag(self.refidx)
        
        self.file1_wav = 1240E-9/(self.omega) #in m
        
        
        arr = np.transpose(np.vstack((self.file1_wav,self.n1, self.k1)))

        #np.savetxt("ref_indx.txt", arr)

      elif self.source == 'experiment':
        
        self.fileName = fileName
        data = np.loadtxt(self.fileName,skiprows=1)
        # Assign columns to variables
        self.file1_wav = data[:, 0]*1E-6 #assuming the experiemtn wavelength is in um
        self.n1 = data[:, 1]
        self.k1 = data[:, 2]
        print("Done Baby")
        # self.eps = self.e0+(self.f/((self.exc**2-self.omega**2)-1j*self.gam*self.omega))
        
        # self.refidx = np.sqrt(self.eps)
        # self.n1 = np.real(self.refidx)
        # self.k1 = np.imag(self.refidx)
        
        #self.file1_wav = 1240E-9/(self.omega) #in m
        #self.file.close
        
        return

      elif self.source == 'none':
        return

      else:
        print('ERROR: Enter the source as "experiment", "theory" or "none". ')
        sys.exit
      
      if draw == 1:
         

          fig, ax = plt.subplots()

          ax.plot(self.file1_wav, self.n1, label='Dataset 1')
          ax.plot(self.file1_wav, self.k1, label='Dataset 2')

          # Set auto scale for both plots
          ax.autoscale(enable=True, axis='both', tight=True)

          ax.set_xlabel('Energy in EV')
          ax.set_ylabel('Refractive Index')
          ax.set_title('Real and Complex r.i.')
          ax.legend()

          plt.show()
          # plt.figure()
          # plt.plot(self.file1_wav,self.n1)
          # plt.show()
          print('Dileep')

      elif draw == 0:
          return
      else:
          print('Enter either 0 or 1')
    
    def expt(self,):

        return
        
    def EM(self,wl = 300, wg = 750, w_step = 1, pol = 1,angl = np.array([0,45]),
            angle_max = 60,angle_step = 1,draw = 0):
            
        self.wavelength = (np.linspace(wl,wg,int(((wg-wl)/w_step)+1)))*1E-9 # in m
        self.ref_n1 = interp1d(self.Reverse(self.file1_wav), self.Reverse(self.n1),kind='cubic')(self.wavelength)
        self.ref_k1 = interp1d(self.Reverse(self.file1_wav), self.Reverse(self.k1),kind='cubic')(self.wavelength)
        
        self.layr3_n_cmplx = self.ref_n1-(1j*self.ref_k1)
        self.layr3_n_cmplx =  self.layr3_n_cmplx.reshape(1,len( self.layr3_n_cmplx)) # to make ths shape (1,301) for default values of len(wavelength)

        self.pol = pol
        self.angl = angl
        self.angle_max = angle_max
        self.angle_step = angle_step

        if draw == 1:
          fig, axs = plt.subplots(2)
          axs[0].plot(self.wavelength,self.ref_n1)
        
          axs[1].plot(self.wavelength,self.ref_k1)
          plt.show()
          
        elif draw == 0:
            return
        else:
          print('Enter either 0 or 1')

        
    
    def DBR(self,Bragg = 555, mode = 10, air_n = 1, DBR_per_up = 2,DBR_per_bot = 2,
              lr1_n = 1.5, lr2_n = 2.5, cav_n = 1.5, lr4_n = 2.5, lr5_n = 1.5, sub_n = 1.5,
              exc_num = 1, exc_thick = 1):
        self.Bragg = Bragg*1E-9
        self.mode = mode
        
        self.air_n = air_n
        self.DBR_per_up = int(DBR_per_up)
        self.DBR_per_bot = int(DBR_per_bot)

        #below two are only used to plot the DBR schamtic. Not for any calulations.
        self.DBR_per_up_for_schematic = DBR_per_up
        self.DBR_per_bot_for_schematic = DBR_per_bot

        self.lr1_n = lr1_n
        self.lr2_n = lr2_n
        self.cav_n = cav_n
        self.lr4_n = lr4_n
        self.lr5_n = lr5_n
        self.sub_n = sub_n
        self.exc_num = exc_num
        self.exc_thick = exc_thick*1E-9 #in m now

        self.cav_layers = 2*self.exc_num+1
        self.air_n_real_mat = self.air_n*np.ones((1,len(self.wavelength)))

        self.sub_mat = self.sub_n*np.ones((1,len(self.wavelength)))
        self.cav_indices = np.vstack((self.cav_n*np.ones((1,len(self.wavelength))), self.layr3_n_cmplx))
        
        self.up_DBR_indx = np.vstack((self.lr1_n*np.ones((1,len(self.wavelength))),self.lr2_n*np.ones((1,len(self.wavelength)))))
        self.lw_DBR_indx = np.vstack((self.lr4_n*np.ones((1,len(self.wavelength))),self.lr5_n*np.ones((1,len(self.wavelength)))))

        if self.exc_num !=0:

          self.indx_mat = np.vstack((self.air_n_real_mat,self.up_DBR_indx,
                    np.vstack((self.cav_indices,)*self.exc_num),
                    self.cav_n*np.ones((1,len(self.wavelength))),
                    self.lw_DBR_indx,self.sub_mat)) #shape is (11,301) for default values
        else:
          self.indx_mat = np.vstack((self.air_n_real_mat,self.up_DBR_indx,
                    self.cav_n*np.ones((1,len(self.wavelength))),
                    self.lw_DBR_indx,self.sub_mat))

        self.thick_layer1 = self.Bragg/(4*self.lr1_n) # in m
        self.thick_layer2 = self.Bragg/(4*self.lr2_n) # in m
        self.thick_layer4 = self.Bragg/(4*self.lr4_n) # in m
        self.thick_layer5 = self.Bragg/(4*self.lr5_n) # in m

        self.tot_cav_thick = self.mode*(self.Bragg)/(2*self.cav_n) #thickness of total cavity inside DBRs
        print('Braggs cavity thickness (no DBR) : ', self.tot_cav_thick)
        #When exc_num is placed, total cavity length (tot_cav_thcik), is divided into 
        #equally (1/(self.exc_num+1)) parts. That's now effective cavity length is 
        # (1/(self.exc_num+1))*self.tot_cav_thick + exciton thickness of one layer * exc_num
        
        #self.cav_layer_thick is 1/3 of total Braggs cavity thickness (i.e. self.tot_cav_thick).
        self.cav_layer_thick = (1/(self.exc_num+1))*self.tot_cav_thick; #d excludes exciton thicknesses; Total thickness is excitons' thickness+tot_cav_thick

        #self.cav_thick_mat is a pair of '1/3 of the Braggs cavity thickness and exciton layer thickenss'
        self.cav_thick_mat = np.vstack((self.cav_layer_thick,self.exc_thick)) #2x1
        print('Original cavity thickness including all the exciton layers (no DBR): ', (self.cav_thick_mat*self.exc_num)+self.cav_layer_thick)
        
        if self.exc_num!=0:

          self.thick_mat = np.vstack((self.thick_layer1,self.thick_layer2,
                  np.vstack((self.cav_thick_mat,)*self.exc_num),
                  self.cav_layer_thick,self.thick_layer4,self.thick_layer5))
        else:
          self.thick_mat = np.vstack((self.thick_layer1,self.thick_layer2,
                  self.cav_layer_thick,self.thick_layer4,self.thick_layer5))
        return self.tot_cav_thick, (self.cav_thick_mat*self.exc_num)+self.cav_layer_thick, self.thick_layer1,self.thick_layer2,self.thick_layer4,self.thick_layer5 
    
    def calc(self,finite_sub_indx = 1.5, finite_sub_thick = 0):


        self.z = 0
        self.angle_set = (np.linspace(-1*self.angle_max,self.angle_max,int(((2*self.angle_max)/self.angle_step)+1)))*np.pi/180
        self.Reflectivity = np.empty((len(self.wavelength),len(self.angle_set)))
        self.Angle = []
        
        

        for angle in tqdm(self.angle_set):
          self.refr_theta_mat = np.empty((0,len(self.wavelength))) #create empty array with fixed number of coulums and row matrices are added later
          self.refr_theta_mat = np.append(self.refr_theta_mat,angle*np.ones((1,len(self.wavelength))),axis = 0)
          for i in np.arange(1,6+self.cav_layers):
            self.refr_theta_mat = np.append(self.refr_theta_mat,np.arcsin((self.indx_mat[[i-1]]*np.sin(self.refr_theta_mat[[i-1]])/self.indx_mat[[i]])),
                                  axis = 0)
            
          self.ad_mat = np.empty((0,len(self.wavelength))) # zeros(length(thick_mat)+2,length(Wavelength));
          for i in np.arange(0,6+self.cav_layers):
            self.ad_mat = np.append(self.ad_mat,self.indx_mat[[i]]*np.cos(self.refr_theta_mat[[i]])*self.ad_freespace,
                                  axis = 0)
        
          
          self.phase_mat = np.empty((0,len(self.wavelength)))
          for i in np.arange(1,5+self.cav_layers): #phase only calculated without air and substrate
                
            self.phase_mat = np.append(self.phase_mat,2*np.pi*self.indx_mat[[i]]*self.thick_mat[[i-1]]*np.cos(self.refr_theta_mat[[i]])/self.wavelength,
                                  axis = 0)
            
          self.ph_m_sze = np.shape(self.phase_mat)
          self.ad_m_sze = np.shape(self.ad_mat)

          for i in np.arange(0,len(self.wavelength)):
            self.M = np.asmatrix([[1,0],[0,1]])

            #k corresponds to number of layers
            #below is for upper DBR calc
            for k in np.array([0,1]):
              self.b = np.asmatrix([np.cos(self.phase_mat[[[k],[i]]]),(np.sin(self.phase_mat[[[k],[i]]])/self.ad_mat[[[k+1],[i]]])*1j]).reshape(1,2)
              self.c = np.asmatrix([(self.ad_mat[[[k+1],[i]]])*np.sin(self.phase_mat[[[k],[i]]])*1j,np.cos(self.phase_mat[[[k],[i]]])]).reshape(1,2)
              self.M = self.M*np.concatenate((self.b,self.c))

              if k == 1:
                self.M = self.M**self.DBR_per_up
                # print("d")
            #below is for cav and exciton calc
            for k in np.arange(2,self.ph_m_sze[0]-2): # here k = 2 for cavity starting layer.
              self.b = np.asmatrix([np.cos(self.phase_mat[[[k],[i]]]),(np.sin(self.phase_mat[[[k],[i]]])/self.ad_mat[[[k+1],[i]]])*1j]).reshape(1,2)
              self.c = np.asmatrix([(self.ad_mat[[[k+1],[i]]])*np.sin(self.phase_mat[[[k],[i]]])*1j,np.cos(self.phase_mat[[[k],[i]]])]).reshape(1,2)
              self.M = self.M*np.concatenate((self.b,self.c))
              # print("dile")

            self.M1 = np.asmatrix([[1,0],[0,1]])
            
            #below is for bottom DBR calc
            for k in np.arange(self.ph_m_sze[0]-2,self.ph_m_sze[0]):
              self.b = np.asmatrix([np.cos(self.phase_mat[[[k],[i]]]),(np.sin(self.phase_mat[[[k],[i]]])/self.ad_mat[[[k+1],[i]]])*1j]).reshape(1,2)
              self.c = np.asmatrix([(self.ad_mat[[[k+1],[i]]])*np.sin(self.phase_mat[[[k],[i]]])*1j,np.cos(self.phase_mat[[[k],[i]]])]).reshape(1,2)
              self.M1 = self.M1*np.concatenate((self.b,self.c))

              if k == self.ph_m_sze[0]-1:
                self.M = self.M*self.M1**self.DBR_per_bot

            #code for finite substrate
            self.finite_sub_indx = finite_sub_indx*np.ones((1,len(self.wavelength))) #same for all wavelengths and angle
            self.finite_sub_thick = finite_sub_thick*np.ones((1,len(self.wavelength)))
            self.finite_sub_ref_angle = np.arcsin((self.indx_mat[-2,:]*np.sin(self.refr_theta_mat[-2,:]))/self.finite_sub_indx) #-2 is taken to access the l5 thickness and r.i.
            self.finite_sub_phase = 2*np.pi*self.finite_sub_indx*self.finite_sub_thick*np.cos(self.finite_sub_ref_angle)/self.wavelength
            self.finite_sub_admtnce = self.finite_sub_indx*np.cos(self.finite_sub_ref_angle)*self.ad_freespace

            self.b = np.asmatrix([np.cos(self.finite_sub_phase[0,i]),(np.sin(self.finite_sub_phase[0,i])/(self.finite_sub_admtnce[0,i]))*1j]).reshape(1,2)
            self.c = np.asmatrix([(self.finite_sub_admtnce[0,i])*np.sin(self.finite_sub_phase[0,i])*1j,np.cos(self.finite_sub_phase[0,i])]).reshape(1,2)
            self.M2 = np.concatenate((self.b,self.c))

            self.M = self.M*self.M2

            self.c = self.M[[[1],[0]]] + self.M[[[1],[1]]]*self.ad_mat[[[self.ad_m_sze[0]-1],[i]]]; #substrate effect comes
            self.b = self.M[[[0],[0]]] + self.M[[[0],[1]]]*self.ad_mat[[[self.ad_m_sze[0]-1],[i]]]; #substrate effect comes
            
            self.denominator = self.ad_mat[[[0],[i]]]*self.b + self.c
            self.numerator = self.ad_mat[[[0],[i]]]*self.b - self.c
            
            self.r = self.numerator/self.denominator
           
            self.Reflect = self.r*np.conj(self.r)
            self.Reflectivity[[[i],[self.z]]] = np.real(self.Reflect)
          
          self.z = 1 + self.z
        
    def plot_reslt(self,ax = None):

      if ax is None:
            fig, ax = plt.subplots()
      else:
            fig = ax.figure

      #extend = [self.angle_set[0]*180/np.pi,self.angle_set[len(self.angle_set)-1]*180/np.pi,1240*1E-9/self.wavelength[len(self.wavelength)-1],1240*1E-9/self.wavelength[0]]
      
      #for plotting in energy 
      # extend = [self.angle_set[0]*180/np.pi,self.angle_set[len(self.angle_set)-1]*180/np.pi,1240*1E-9/self.wavelength[len(self.wavelength)-1],1240*1E-9/self.wavelength[0]]
      # img = ax.imshow(self.Reflectivity,extent = extend,aspect = 'auto')
      # ax.set_xlabel('Angle(degree)')
      # ax.set_ylabel('Photon Energy (eV)')

      #for plottingin wavelenth. extend is setting x y limits for axes
      extend = [self.angle_set[0]*180/np.pi,self.angle_set[len(self.angle_set)-1]*180/np.pi,self.wavelength[len(self.wavelength)-1],self.wavelength[0]]
      img = ax.imshow(self.Reflectivity,extent = extend,aspect = 'auto')
      ax.set_xlabel('Angle(degree)')
      ax.set_ylabel('Wavelength (um)')

      #colorbar axis
      cbar_ax = fig.add_axes([0.025, 0.025, 0.025, 0.25])  # <-- added this line
      fig.colorbar(img, cax=cbar_ax)  # <-- modified this line
      #fig.colorbar(img, )


      return fig, ax 
      
      
      
    
    
    def plot_0Deg(self, ax = None, given_inputs = [0]):
      
      if ax is None:
            fig, ax = plt.subplots()
      else:
            fig = ax.figure
      
      
      
      #aa, bb = self.Reflectivity.shape
      for given_input in given_inputs:
        index = np.abs(self.angle_set*180/np.pi - given_input).argmin()
        #closest_actual_value = self.angle_set[index]
      

        self.Deg0 = self.Reflectivity[:,index]

        ax.plot(self.wavelength*1E6,self.Deg0, label=f'Angle: {self.angle_set[index]*180/np.pi}')

      #ax.plot(self.wavelength*1E6,self.Deg0)
      ax.legend(loc = "upper right")
      #ax.set_ylim(ymin=-0.1, ymax = 0.1)
      ax.set_xlabel('Wavelength (um)')
      ax.set_ylabel('Reflectivity (a.u.)')
      return fig, ax

     
    
    def save_text(self):

      aa, bb = self.Reflectivity.shape
      Deg0 = self.Reflectivity[:,bb//2]
      arr = np.transpose(np.vstack((self.wavelength,Deg0)))

      np.savetxt("Deg0.txt", arr)