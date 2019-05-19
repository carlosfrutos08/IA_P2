from scipy.spatial.distance import cdist as dist
"""

@author: ramon, bojana
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from sklearn.decomposition import PCA
import math as m


def NIUs():
    return 111111, 1111112, 1111113
    
def distance(X,C):
    """@brief   Calculates the distance between each pixcel and each centroid 

    @param  X  numpy array PxD 1st set of data points (usually data points)
    @param  C  numpy array KxD 2nd set of data points (usually cluster centroids points)

    @return dist: PxK numpy array position ij is the distance between the 
    	i-th point of the first set an the j-th point of the second set
    """
    C=np.array(C)
    if len(C.shape) == 1:
        C=[C]       
    return dist(X,C)


  

#########################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#########################################################
    

class KMeans():
    
    def __init__(self, X, K, options=None):
        """@brief   Constructor of KMeans class
        
        @param  X   LIST    input data
        @param  K   INT     number of centroids
        @param  options DICT dctionary with options
        """

        self._init_X(X)                                    # LIST data coordinates
        self._init_options(options)                        # DICT options
        self._init_rest(K)                                 # Initializes de rest of the object
        
#############################################################
##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
#############################################################

        
    def _init_X(self, X):
        """@brief Initialization of all pixels
        
        @param  X   LIST    list of all pixel values. Usually it will be a numpy 
                            array containing an image NxMx3

        sets X an as an array of data in vector form (PxD  where P=N*M and D=3 in the above example)
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        self.X = np.zeros((2,X.shape[0]))
        if X.ndim > 2:
            self.X=np.reshape(X,(-1,X.shape[2]))
        else:
            self.X=X[:]
       # print("Matrix x incialitzed",self.X)     


            
    def _init_options(self, options):
        """@brief Initialization of options in case some fields are left undefined
        
        @param  options DICT dctionary with options

			sets de options parameters
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'Fisher'

        self.options = options
        
#############################################################
##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
#############################################################

        
    def _init_rest(self, K):
        """@brief   Initialization of the remainig data in the class.
        
        @param  options DICT dctionary with options
        """
        self.K = K                                             # INT number of clusters
        if self.K>0:
            self._init_centroids()                             # LIST centroids coordinates
            self.old_centroids = np.empty_like(self.centroids) # LIST coordinates of centroids from previous iteration
            self.clusters = np.zeros(len(self.X))              # LIST list that assignes each element of X into a cluster
            self._cluster_points()                             # sets the first cluster assignation
        self.num_iter = 0                                      # INT current iteration
            
#############################################################
##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
#############################################################


    def _init_centroids(self):
        """@brief Initialization of centroids
        depends on self.options['km_init']
        """
        if self.options['km_init'] == 'first':
            c = []
            c = np.array(c)
            c = [np.append(c, self.X[0], axis=0)]
            c = np.array(c)
            i=1     
            # Considerarem els primers pixels com a centroides
            while (c.shape[0]<self.K):
                c2=[self.X[i]]
                if not np.array_equal(c, c2):
                    c=np.append(c, c2, axis=0)
                    #print aux
                i+=1
            self.centroids = c
           
        elif self.options['km_init'] == 'random':        # Considerarem pixels aleatoris com a centroids
            self.centroids = np.random.rand(self.K,self.X.shape[1])
        elif self.options ['km_init'] ==  'custom':
            self.centroids = np.zeros((self.K,self.X.shape[1]))
            print (self.centroids)    
            var=self.K-1
            c=255.0
            print(var)
            for k in range(self.K): self.centroids[k,:] = k*255/(self.K-1)
            print (self.centroids)
            
        
            
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        
      
        
        
    def _cluster_points(self):
        """@brief   Calculates the closest centroid of all points in X
        """
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        #print("x:",self.X)
        #print("centroids:",self.centroids)
        distances = distance(self.X, self.centroids)
        #print ("distances:",distances)
        self.clusters = np.argmin(distances, axis=1)
        
    
        
        

        
    def _get_centroids(self):
        """@brief   Calculates coordinates of centroids based on the coordinates 
                    of all the points assigned to the centroid
        """
        self.old_centroids=self.centroids
        temp_centroids=np.zeros((self.K,self.X.shape[1]))
        for i in range(self.K):
            pixels_cluster=0   
            for j in range(len(self.clusters)):           
                if(i == self.clusters[j]):
                    temp_centroids[i]+=self.X[j]
                    pixels_cluster+=1
            if(pixels_cluster != 0):        
               self.centroids[i]=(temp_centroids[i]/pixels_cluster)
           
                
                
                
            
            
           
        

#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################


       
                
    def _converges(self):
        """@brief   Checks if there is a difference between current and old centroids
        """
        return np.isclose(self.old_centroids, self.centroids, self.options['tolerance']).all()
                
                
       
       
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
        
        
    def _iterate(self, show_first_time=True):
        """@brief   One iteration of K-Means algorithm. This method should 
                    reassigne all the points from X to their closest centroids
                    and based on that, calculate the new position of centroids.
        """
        self.num_iter += 1
        self._cluster_points()
        self._get_centroids()
        if self.options['verbose']:
            self.plot(show_first_time)


    def run(self):
        """@brief   Runs K-Means algorithm until it converges or until the number
                    of iterations is smaller than the maximum number of iterations.=
        """
        if self.K==0:
            self.bestK()
            return        
        
        self._iterate(True)
        if self.options['max_iter'] > self.num_iter:
            while not self._converges() :
                self._iterate(False)
      
      
    def bestK(self):
        """@brief   Runs K-Means multiple times to find the best K for the current 
                    data given the 'fitting' method. In cas of Fisher elbow method 
                    is recommended.
                    
                    at the end, self.centroids and self.clusters contains the 
                    information for the best K. NO need to rerun KMeans.
           @return B is the best K found.
        """
        if self.options['fitting'].lower() == 'fisher':
            
            bk=0
            x=0
            tmp_fit=[]
            tmp=0.0
            for i in range(3, 10):
                self._init_rest(i)
                print(i)
                x=0
                print("lista:",tmp_fit)
                while(x < 10):
                     self.run()
                     tmp+= self.fitting()
                     print(tmp)
                     x+=1    
                tmp_fit.append(tmp/10)
                
            min_fit=min(tmp_fit)
            bk=tmp_fit.index(min_fit)+3          
            print("bk:",bk)        
            return bk
        else:
            self._init_rest(4)
            self.run()
            return 4
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
       
        
    def fitting(self):
        """@brief  return a value describing how well the current kmeans fits the data
        """
        if self.options['fitting'].lower() == 'fisher':
            within_variance = 0
            distance_list = distance(self.X, self.centroids)
         
            for K in range(self.K):
                sum_distance_cluster = 0
                pixels_cluster = 0
                for pixel in range(len(self.clusters)):
                    if self.clusters[pixel] == K:
                        sum_distance_cluster += distance_list[pixel][K]
                        pixels_cluster += 1
                if(pixels_cluster !=0):        
                    within_variance += sum_distance_cluster/pixels_cluster

            between_variance = 0
            mean=np.mean(self.X,axis=0)
            x=np.array(mean)
            x=x.reshape(-1,self.centroids.shape[1])
            #print("x",x)
            #print("self.centroids",self.centroids)
            meanvscentroids=distance(x,self.centroids)
            #print("meanvscentroids",meanvscentroids)
            #print(len(self.centroids))
            for i in range(len(self.centroids)):
                between_variance+=meanvscentroids[0][i]
                
                
            between_variance/self.K
            print(within_variance/between_variance)
          
            return within_variance/between_variance
        else:
            return np.random.rand(1)
        
#######################################################
##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
##  AND CHANGE FOR YOUR OWN CODE
#######################################################
  


    def plot(self, first_time=True):
        """@brief   Plots the results
        """

        #markersshape = 'ov^<>1234sp*hH+xDd'	
        markerscolor = 'bgrcmybgrcmybgrcmyk'
        if first_time:
            plt.gcf().add_subplot(111, projection='3d')
            plt.ion()
            plt.show()

        if self.X.shape[1]>3:
            if not hasattr(self, 'pca'):
                self.pca = PCA(n_components=3)
                self.pca.fit(self.X)
            Xt = self.pca.transform(self.X)
            Ct = self.pca.transform(self.centroids)
        else:
            Xt=self.X
            Ct=self.centroids

        for k in range(self.K):
            plt.gca().plot(Xt[self.clusters==k,0], Xt[self.clusters==k,1], Xt[self.clusters==k,2], '.'+markerscolor[k])
            plt.gca().plot(Ct[k,0:1], Ct[k,1:2], Ct[k,2:3], 'o'+'k',markersize=12)

        if first_time:
            plt.xlabel('dim 1')
            plt.ylabel('dim 2')
            plt.gca().set_zlabel('dim 3')
        plt.draw()
        plt.pause(0.01)