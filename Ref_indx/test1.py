def DBRplot(self,ax = None):
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
      for i in range(self.DBR_per_up):
        rectangle1 = plt.Rectangle((0,-y1), 50, -self.thick_layer1, fc=self.c_map(self.lr1_n),ec="black")
        rectangle2 = plt.Rectangle((0,-y1-self.thick_layer1), 50, -self.thick_layer2, fc=self.c_map(self.lr2_n),ec="black")


        ax.add_patch(rectangle1)
        ax.add_patch(rectangle2)

        ax.annotate(str(self.lr1_n), (55, -y1-self.thick_layer1/2), color='b', weight='bold', 
                 fontsize=6, ha='center', va='center')
        ax.annotate(str(self.lr2_n), (55, -y1-self.thick_layer1-self.thick_layer2/2), color='b', weight='bold', 
                 fontsize=6, ha='center', va='center')
        y1 = y1+self.thick_layer1+self.thick_layer2

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

      
      for i in range(self.DBR_per_bot):
        rectangle1 = plt.Rectangle((0,-y1), 50, -self.thick_layer4, fc=self.c_map(self.lr4_n),ec="black")
        rectangle2 = plt.Rectangle((0,-y1-self.thick_layer4), 50, -self.thick_layer5, fc=self.c_map(self.lr5_n),ec="black")

        ax.add_patch(rectangle1)
        ax.add_patch(rectangle2)

        ax.annotate(str(self.lr4_n), (55, -y1-self.thick_layer4/2), color='b', weight='bold', 
                 fontsize=6, ha='center', va='center')
        ax.annotate(str(self.lr5_n), (55, -y1-self.thick_layer4-self.thick_layer5/2), color='b', weight='bold', 
                 fontsize=6, ha='center', va='center')

        y1 = y1+self.thick_layer4+self.thick_layer5

        # plt.axis('scaled')
      ax.annotate(str(self.sub_n), (25, -y1-self.thick_layer5/2), color='b', weight='bold', 
                 fontsize=6, ha='center', va='center')
      ax.autoscale()
      ax.set_xlim(0,70)

      return fig, ax