###############################            By BOUSBA Abdellah            #############################


######################################################################################################


						#usage : type  "python ships.py -h" for help 
						
						
######################################################################################################

import numpy as np 
import matplotlib.pyplot as plt
import argparse
import time


################################    Script parameters    ##############################################


parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, help="Size of the play zone (matrix)", default=10)
parser.add_argument("-nbBat", type=int, help="Number of ships", default=5)
parser.add_argument("-affichage",  action="count", help="show grille evry time a ship is hit", default=0)
parser.add_argument("-gen","--gen", action="count", help="generate a random grille of 5 ships", default=0)
parser.add_argument("-play1", "--playAlea", action="count", help="play with version aleatoire", default=0)
parser.add_argument("-play2", "--playHeur", action="count", help="play with version heurestique", default=0)
parser.add_argument("-play3", "--playSimple", action="count", help="play with version proba simple", default=0)
parser.add_argument("-var", "--afficherVariance", action="count", help="print variance of each version", default=0)
parser.add_argument("-plot", "--dessinerCourbe", action="count", help="plot repartition function for each version", default=0)
parser.add_argument("-t", "--time", action="count", help="print time used of each function", default=0)
args = parser.parse_args()


N=args.N #Size of the play zone (matrix)
nbBat=args.nbBat # Number of ships
affichage = (args.affichage == 1)
assert N>0
assert nbBat>0

##############################     Classes      ###############################

class Bateau:
    
    def __init__(self, longeur, position=(0,0), direction=1):
        self.longeur=longeur
        self.position=position
        self.direction=direction
        self.tab = [longeur] * longeur	# pour avoir l'information sur chaque case du bateau (touché ou pas encore)
        
    def isDead(self): #savoir si le bateau est tombé ou pas encore
        for i in range(self.longeur):
            if(self.tab[i] != -1):
                return False
        return True

    def __eq__(self,b):	# redefinition de equals
         return b != None and b.longeur == self.longeur and b.position == self.position and b.direction == self.direction
        
    def __lt__(self,b): #redefinition de lt pour avoir un tri croissant en utilisant sort sur une liste des bateaux
        if(b.longeur == self.longeur):
            if(b.position[0] ==self.position[0]):
                if(b.position[1] ==self.position[1]):
                    return b.direction > self.direction
                else:
                    return b.position[1] > self.position[1]
            else:
                return b.position[0] > self.position[0]
        else:
            return b.longeur > self.longeur
			
			
class Grille:
    
    def __init__(self, N, liste=[],mat=np.asarray([[0] * N for _ in range(N)])):
        self.N=N
        self.liste=liste.copy() # liste des bateau dans la grille
        self.mat=mat
        
        
    def __eq__(self, g):	# redefinition de equals pour avoir une meilleir complixité on utilise l'egalité entre les listes des bateaux
        if(self.N != g.N or len(self.liste) != len(g.liste)):
            return False
        self.liste.sort()
        g.liste.sort()
        return self.liste == g.liste     
     
    def put(self,pos,b):	# mettre le int b dans la matrice, (exemple: -1 sur la case ou on a tiré)
        out = Grille(self.N,self.liste,self.mat.copy())
        out.mat[pos[0]][pos[1]] = b
        return out
    
    def peut_placer(self, b):
        if b.direction == 1:
            for i in range(b.longeur):
                if b.position[0]+i >= len(self.mat) or self.mat[b.position[0]+i][b.position[1]] != 0:
                    return False;
        if b.direction == 2:
            for i in range(b.longeur):
                if b.position[1]+i >= len(self.mat) or self.mat[b.position[0]][b.position[1]+i] != 0:
                    return False;
        return True   

   
        
    def place(self, b):
        out = Grille(self.N,self.liste,self.mat.copy())
        if b.direction == 1:
            for i in range(b.longeur):
                out.mat[b.position[0]+i][b.position[1]] = b.tab[i]
            out.liste.append(b) # NB :on ajoute chaque bateau placer a la liste des bateaux de grille
        if b.direction == 2:
            for i in range(b.longeur):
                out.mat[b.position[0]][b.position[1]+i] = b.tab[i]
            out.liste.append(b)
        return out  
    
    
    def place_alea(self, longeur):
        while(True):
		# pour eviter de tirer la meme case deux fois on utilise une methode sans remise
		# on construit l'ensemble des cas possible et on choisir aleatoirement une position
		# apres chaque tentative de placement on supprime la position utilisé
            indices = [(i,j) for i in range(self.N) for j in range(self.N)] 
            x = np.random.randint(0,len(indices))
            b=Bateau(longeur,indices[x],np.random.randint(1,3))
            if(self.peut_placer(b)):
                return self.place(b)
            indices.pop(x)
            if(len(indices) == 0):
                raise("No more positions left")
            
            
    def affiche(self):
        for (j,i),label in np.ndenumerate(self.mat):
            plt.text(i,j,label,ha='center',va='center')
        plt.imshow(self.mat)
        plt.show()
		
		
class Bataille:
        
    def __init__(self, N):
        self.N=N
        self.liste=[Bateau(2),Bateau(3),Bateau(3),Bateau(4),Bateau(5)] # dans le context du projet on utilise la liste suivante
        self.grille=genere_grille(self.liste,N)
    
    
    def joue(self,position):
        out = False
        bat=None
        if self.grille.mat[position[0],position[1]] != 0: #si on est sur un bateau
            x=len(self.grille.liste)
            for b in self.grille.liste: # on cherche le bateau qu'on a touché pour le mettre a jour
                if b.direction == 2:
                    if b.position[0] == position[0]:
                        if b.position[1] <= position[1] and b.position[1]+b.longeur > position[1]:
                            b.tab[position[1]-b.position[1]] = -1
                            self.grille.mat[position[0]][position[1]] = -1
                            out=True
                if b.direction == 1:
                    if b.position[1] == position[1]:
                        if b.position[0] <= position[0] and b.position[0]+b.longeur > position[0]:
                            b.tab[position[0]-b.position[0]] = -1
                            self.grille.mat[position[0]][position[1]] = -1
                            out=True
            i=0
            for _ in range(len(self.grille.liste)): # on supprime le bateau qui est tombé et on le retourne 
                if self.grille.liste[i].isDead() == True:
                    bat = self.grille.liste.pop(i)
                    i-=1
                i+=1
        return out,bat
                    
        
    def victoire(self):
        return len(self.grille.liste) == 0
    
    def reset(self):
        self.grille=genere_grille(self.liste,self.N)
	
	
class Joueur:
        
    def __init__(self, strat, N=N):
        self.N=N
        self.strat = strat # la strategie ( version du jeu )
    
    def setStrat(self,x):
        self.strat = x
        
    def jouer(self):
        if self.strat == 1:
            return self.jouer_alea()
        if self.strat == 2:
            return self.jouer_heuristique()
        if self.strat == 3:
            return self.jouer_probaSimple()
        
    def jouer_alea(self):
        B = Bataille(self.N)
        cpt=0
        indices = [(i,j) for i in range(B.N) for j in range(B.N)]
        win = False
		# on utilise la meme methode sans remise que place_alea pour tirer sur l'ensembe des cases de la grille aleatoirement
        while(win == False and len(indices) > 0): 
            x = np.random.randint(0,len(indices))
            if(B.joue(indices[x])[0]):
                if affichage:
                    B.grille.affiche()
            cpt+=1
            if(B.victoire()):
                win = True
            indices.pop(x)
        return cpt, win
    

    
        
    def jouer_heuristique(self): # intialitation de la fonction recursive 
        B = Bataille(self.N)
        indices = [(i,j) for i in range(B.N) for j in range(B.N)]
        return self.jouer_heuristique_rec(B,indices,(-1,-1),0)
    
    
    def jouer_heuristique_rec(self,B,indices,pos,cpt):
        if(B.victoire()): # on a gagner
            return cpt, True
        if len(indices) == 0: # plus de cases possible (unreachable, only for debug purposes)
            return cpt, False
        if pos == (-1,-1):# la case precedante etait une case vide 
            x = np.random.randint(0,len(indices)) # on choisit une case aleatoirement de l'ensemble des cases restantes
            temp_pos = indices[x]
            indices.pop(x)
            boolean, bateau = B.joue(temp_pos)
            if(boolean == True): # on indique qu'on a toucher une case 
                if affichage: 
                    B.grille.affiche()
                return self.jouer_heuristique_rec(B,indices,temp_pos,cpt+1)
            else:	# sinon appel avec case vide
                return self.jouer_heuristique_rec(B,indices,(-1,-1),cpt+1)
        else: # la case precedante etait une case d'un bateau
            new_pos =[(pos[0]+1,pos[1]),(pos[0],pos[1]+1),(pos[0]-1,pos[1]),(pos[0],pos[1]-1)]
            ok=4
            for i in range(len(new_pos)): # on explore les cases connexes 
                if new_pos[i] in indices: # avec verification qu'on a pas encore tirer sur chaque case 
                    indices.remove(new_pos[i]) #sasn remise
                    cpt+=1
                    boolean, bateau = B.joue(new_pos[i])
                    if(boolean == True):
                        if affichage: 
                            B.grille.affiche()
                        ok-=1
                        return self.jouer_heuristique_rec(B,indices,new_pos[i],cpt)
            if ok == 4: # si on toucher aucune case on appel avec une case vide 
                return self.jouer_heuristique_rec(B,indices,(-1,-1),cpt)
        
        
    def jouer_probaSimple(self): #BEEEEEEEEESTTTT VERSIOOON
        B = Bataille(self.N)
        indices = [(i,j) for i in range(B.N) for j in range(B.N)]
        bateau_left = B.grille.liste
        bateau_left.sort() # tri en ordre croissant
        g = Grille(N)
        proba, proba_index = get_proba(bateau_left[0].longeur,g) # on calcule la probabilité qu'un bateau soit dans chaque case et les indices trié de chaque case en ordre decroissant
        win = False
        cpt=0
        while(win == False and len(indices) > 0): 
            l = bateau_left[0].longeur # la longeur du bateau min
            x = proba_index[0] # la position de la case avec la probabilité max
            proba_index.remove(x) #sans remise
            if x in indices: # si on a pas encore tirer sur cette case 
                indices.remove(x)# sans remise
                cpt+=1
                boolean, bateau = B.joue(x)
                if(bateau != None): # si un bateau est detruit
                    g = g.place(bateau)	# on permet au joueur a savoir qu'il a fait tomber un bateau

                if(boolean == False):  # si on a tirer sur une case vide on place -1 car c'est impossible d'avoir un bateau la
                    g = g.put((x[0],x[1]),-1)
                if(boolean == True): # si on a toucher un bateau on explore les cases connexes
                    if affichage: 
                        B.grille.affiche()
                    new_pos =[(x[0]+1,x[1]),(x[0],x[1]+1),(x[0]-1,x[1]),(x[0],x[1]-1)]
                    proba_pos = []
                    for pos in new_pos:
                        if pos in indices:
                            proba_pos.append(proba[pos[0]][pos[1]])
                    new_pos = [new_pos[i] for i in (-np.asarray(proba_pos)).argsort()] # on explore les cases connexe par ordre decroissant des probabilité
                    for pos in new_pos:
                        if pos in indices:
                            indices.remove(pos)
                            cpt+=1
                            boolean, bateau = B.joue(pos)
                            if(boolean == True):
                                if affichage: 
                                    B.grille.affiche()
                                break
                            if(bateau != None):
                                g = g.place(bateau)
                            if(boolean == False):
                                g = g.put(pos,-1)

                proba, proba_index = get_proba(l,g) # on recalcule les propabilités
                                    
            
            if(B.victoire()):
                win = True
            
            for i in range(B.N): # on supprime les cases qui on une probabilté 0 
                for j in range(B.N):
                    if proba[i][j] == 0:
                        if (i,j) in indices:
                            indices.remove((i,j))

        return cpt, win





 

###############################################################################



#############################       tools     ################################

def genere_grille(liste,N):
    g = Grille(N)
    for i in range(len(liste)):
        g = g.place_alea(liste[i].longeur)
    return g

	
def denombre_bateau_vide(longeur):
    return (N-longeur+1)*N*2
	
	
def denombre_liste_bateau(liste,grille,indice): # on utilise une fonction recursive
    if indice == len(liste)-1: # si c'est le dernier bateau 
        x=0
        for i in range(grille.N):
            for j in range(grille.N):
                if grille.peut_placer(Bateau(liste[indice].longeur,(i,j),1)):
                    x+=1
                if grille.peut_placer(Bateau(liste[indice].longeur,(i,j),2)):
                    x+=1
        return x #on calcule les cas possible ou on peut le placer et on la retourne 
    if indice < len(liste)-1: # si on a encore des bateau 
        out=0
        for i in range(grille.N):	#pour chaque bateau restant 
            for j in range(grille.N):
                if grille.peut_placer(Bateau(liste[indice].longeur,(i,j),1)): # si on peut le placer 
                    new = grille.place(Bateau(liste[indice].longeur,(i,j),1)) # on le place sur notre grille et on appel la fonction pour le prochaine bateau
                    out+=denombre_liste_bateau(liste,new,indice+1) # on fait la somme des resultat
                if grille.peut_placer(Bateau(liste[indice].longeur,(i,j),2)):
                    new = grille.place(Bateau(liste[indice].longeur,(i,j),2))
                    out+=denombre_liste_bateau(liste,new,indice+1)
        return out
	#Pour tester cette fonction manuellement : exemple : denombre_liste_bateau([Bateau(2),Bateau(3)],Grille(N),0)
		
		
def nb_grille_genere_alea(g):
    new = genere_grille(g.liste,g.N) # on genere une grille
    out=1
    while g != new: # tantque les grilles ne sont pas egeux
        new = genere_grille(g.liste,g.N) # on regenere 
        out+=1 #et compte le nombre de grille genere
    return out
	#Pour tester cette fonction manuellement : exemple : nb_grille_genere_alea(genere_grille([Bateau(2),Bateau(3)],N))
	

def denombre_liste_bateau2(liste,nb): # pour nb iteration on compte le nombre totale de grille genere  
    g = genere_grille(liste,N)
    cpt=0
    for _ in range(nb):
        cpt+=nb_grille_genere_alea(g)
    return cpt/nb #dans cpt experiance on a nb grille alors le nombre de grille possible est cpt/nb 
	#Pour tester cette fonction manuellement : exemple : denombre_liste_bateau2([Bateau(2),Bateau(3)],10)
	

	
def get_proba(l,g):
    out = np.asarray([[0] * N for _ in range(N)])
    for i in range(g.N):
        for j in range(g.N): # pour chaque case 
            b1 = Bateau(l,(i,j),1) # vertical
            b2 = Bateau(l,(i,j),2) #horizontal
            if g.peut_placer(b1):
                for k in range(l):
                    out[b1.position[0]+k][b1.position[1]]+=1 # on incremante le nombre possible des bateau pour chaque case ou on peut placer le bateau
            if g.peut_placer(b2):
                for k in range(l):
                    out[b2.position[0]][b2.position[1]+k]+=1
    out = out/denombre_bateau_vide(l) #on divise par le nombre totale de configuration possible
    return out , list(map(tuple,(np.dstack(np.unravel_index(np.argsort((-out).ravel()), (N, N)))[0]))) # on cacule aussi les indices triés par ordre decroissant des propabilité 


def get_varience(J,nb):
    proba = np.asarray([0] * N*N)
    cpt=0
    for i in range(nb): #on joue pour nb parties 
        x = J.jouer()[0]
        proba[x-1]+=1 #pour chaque nombre de tire utilisé pour gagner on incremante
    proba = proba/nb # calculer la probabilité en supposant que les cases sont equiprobable
    out=0
    for i in range(100): # caulcule de la variance 
        out+=(i+1)*proba[i]
    return proba, out

	
def plot_repartition(J,nb,ax): # on donne le ax pour dessiner plusieur courbe dans le meme plot ( ajouter plt.show() apres l'appel de la fonction )
    name = "unknown"
	#choisir le nom de la courbe
    if J.strat == 1:
        name = "version_aleatoire"
    if J.strat == 2:
        name = "version_heuristique"
    if J.strat == 3:
        name = "version_probaSimple"
        
    proba, var = get_varience(J,nb)
    y = [0] * N*N
    cpt=0
    for i in range(N*N): # calculer la distribution (repartition)
        cpt+=proba[i]
        y[i] = cpt
        
    x = [i+1 for i in range(100)]

    ax.step(x,y,label=name)
    ax.axhline(y=1, color='r', linestyle='-')
    ax.set_xlabel('nombre de tires')
    ax.set_ylabel('probabilité')
    ax.legend(loc='best')

	
	
##################################    main      #######################################



 
if args.gen:
    t0 = time.time()
    g = genere_grille([Bateau(2),Bateau(3),Bateau(3),Bateau(4),Bateau(5)],N)
    g.affiche()
    t1 = time.time()
    tGen = t1-t0
    if args.time:
        print("time used is %f s" %(tGen))	
	
if args.playAlea:
    t0 = time.time()
    nb, win = Joueur(1).jouer()
    if win:
        print("Player \"version_aleatoire\" won in : %d tries " %(nb))
    else:
        print("No more moves")
    t1 = time.time()
    tAlea = t1-t0
    if args.time:
        print("time used is %f s" %(tAlea))
	
	
if args.playHeur:
    t0 = time.time()
    nb, win = Joueur(2).jouer()
    if win:
        print("Player \"version_heuristique\" won in : %d tries " %(nb))
    else:
        print("No more moves")
    t1 = time.time()
    tHeur = t1-t0
    if args.time:
        print("time used is %f s" %(tHeur))
	
	
if args.playSimple:
    t0 = time.time()
    nb, win = Joueur(3).jouer()
    if win:
        print("Player \"version_probasimple\" won in : %d tries " %(nb))
    else:
        print("No more moves")
    t1 = time.time()
    tSimple = t1-t0
    if args.time:
        print("time used is %f s" %(tSimple))
		
		
if args.afficherVariance:
    t0 = time.time()    
    proba, var = get_varience(Joueur(1),1000)
    print("Variance \"version_aleatoire\" : %f " %(var))
    t1 = time.time()
    temp = t1-t0
    if args.time:
        print("time used is %f s" %(temp))
    t0 = time.time()    
    proba, var = get_varience(Joueur(2),1000)
    print("Variance \"version_heuristique\" : %f " %(var))
    t1 = time.time()
    temp = t1-t0
    if args.time:
        print("time used is %f s" %(temp))
    t0 = time.time()    
    print("version_probasimple may take a while ....")
    proba, var = get_varience(Joueur(3),1000)
    print("Variance \"version_probasimple\" : %f " %(var))
    t1 = time.time()
    temp = t1-t0
    if args.time:
        print("time used is %f s" %(temp))	
	
	
if args.dessinerCourbe:
    fig, ax = plt.subplots(figsize=(12, 8))
    t0 = time.time()
    plot_repartition(Joueur(1),1000,ax)
    t1 = time.time()
    temp = t1-t0
    if args.time:
        print("time used for alea is %f s" %(temp))
    t0 = time.time()
    plot_repartition(Joueur(2),1000,ax)
    t1 = time.time()
    temp = t1-t0
    if args.time:
        print("time used for heur is %f s" %(temp))
    t0 = time.time()
    print("version_probasimple may take a while ....")
    plot_repartition(Joueur(3),1000,ax)
    t1 = time.time()
    temp = t1-t0
    if args.time:
        print("time used for simple is %f s" %(temp))		
    plt.show()
		


