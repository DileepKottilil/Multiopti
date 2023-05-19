def DBRplot(self, ax = None):
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
      top_full_pairs = int(self.DBR_per_up)
      top_extra_layer = self.DBR_per_up - top_full_pairs

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

      bottom_full_pairs = int(self.DBR_per_bot)
      bottom_extra_layer = self.DBR_per_bot - bottom_full_pairs

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

      # Final layers and annotation...

      return fig, ax