#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:02:57 2018

@author: john
"""

import numpy as np

#used by environments and skel holders
#class to hold reference for all contact info for a particular skeleton's body node
#needs to be held in a collection indexed by body node, so that it doesn't get overwritten
#this class is a bit weird - doesn't have body node set in ctorbecause used in a default dict               
class contactInfo():
    def __init__(self) :
        self.ttlfrc = np.zeros(3)
        self.COPloc = np.zeros(3)
        #np 3 dim arrays for point location and force value
        self.cntctPt = list()
        self.cntctFrc = list()
        self.colBodies= list()
        self.bodyTorques = list()
        self.ttlBdyTrque = None
        self.body = None
        
    
    #if there's a contact with this body
    #thisBody is this node's body node - should not be able to change - one contact info for each body node
    #ctrTauPt is location around which forces are applied to calc torque
    def addContact(self, contact, thisBody, otrBody, ctrTauPt):
        if(None == self.body) :
            self.body = thisBody
            self.skel = thisBody.skeleton
        elif (self.skel.name != thisBody.skeleton.name) or (self.body.name != thisBody.name):
            print('Error in contactInfo:addContact : attempting to reassign from skel {} body {} to new skel {} body {}'.format(self.skel.name, self.body.name,thisBody.skeleton.name, thisBody.name))
        self.colBodies.append(otrBody)
        cp = np.copy(contact.point)
        cf = np.copy(contact.force)
        self.cntctPt.append(cp)
        self.cntctFrc.append(cf)        
        #calc body torques from all contacts -> Jtrans@cntctpt(in body frame) * fcntct
        JAtPt = self.body.linear_jacobian(offset=self.body.to_local(cp))
        bdyTrque = np.transpose(JAtPt).dot(cf)
        self.bodyTorques.append(bdyTrque)
        
        self.setCopLoc(ctrTauPt)
    
    #only call when all body torques are calced
    def calcTtlBdyTrqs(self):
        self.ttlBdyTrque = np.zeros(np.size(self.skel.q,axis=0))
        for bt in self.bodyTorques :
            self.ttlBdyTrque += bt
        return self.ttlBdyTrque
    
    #recalculate average location of contacts - called internally
    def setCopLoc(self, ctrTauPt):
        COP_ttlFrc = np.zeros(3)        
        ttlTau = np.zeros(3)
        COPval = np.zeros(3)
        numPts = len(self.cntctPt)
        
        for i in range(numPts):
            COP_ttlFrc += self.cntctFrc[i]
            #displacement vector from torque pivot
            rVec = self.cntctPt[i] - ctrTauPt
            ttlTau += np.cross(rVec,self.cntctFrc[i])
        #coploc is location of cop in world coords
        #force coploc to be on ground -> coploc[1] = 0
        COPvalTmp = np.cross(COP_ttlFrc,ttlTau)/np.dot(COP_ttlFrc,COP_ttlFrc)#+t*COP_ttlFrc #is a solution for all t.
        #solve for t by finding frc[1]*t - COPval[1]=0 since we want COP val to lie on y=0 plane
        #solve for t by finding frc[1]*t - COPval[1]=0 since we want COP val to lie on y=0 plane
        try:
            t = -float(COPvalTmp[1])/COP_ttlFrc[1]
            COPval = COPvalTmp + t*COP_ttlFrc
        except ZeroDivisionError:
            COPval = COPvalTmp    
        
        #COPval[0] = ttlTau[2]/COP_ttlFrc[1] #+ ctrTauPt[0]
        #COPval[2] = (ttlTau[1]+ (COP_ttlFrc[2] * self.COPloc[0]))/COP_ttlFrc[0] #+ ctrTauPt[2]        
        
        self.ttlfrc = COP_ttlFrc
        self.COPloc = COPval + ctrTauPt
    
    #for optimization just want jacobian for each body
    def getCntctJacob(self):
        #want jacobian expressed in world frame, since forces are expressed in world frame
        Jtrans = np.transpose(self.body.world_jacobian(offset=self.body.to_local(self.COPloc)))
        return Jtrans        
        
    #This will get the contact force in generalized coordinates
    #using J(self.COPloc)_transpose * self.ttlfrc
    def getCntctTauGen(self):
        #want jacobian expressed in world frame, since forces are expressed in world frame
        Jtrans = self.getCntctJacob()
        #Jtrans is dofs x 6 : first 3 of 6 are angular components, 2nd 3 of 6 are linear
        frcVec = np.zeros(6)
        #no angular component
        frcVec[3:]=self.ttlfrc
        #numpy matrix * vector -> vector
        return Jtrans.dot(frcVec)