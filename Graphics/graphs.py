#! /usr/bin/env python3
from matplotlib import pyplot as plt
import numpy as np

## This program should be used for creating plots of some relevant results

class Mitigation:
    def __init__(self, acc_main_class, acc_second_class, x, fig_title, titles):
        self.accuracy_main_class = acc_main_class
        self.accuracy_second_class = acc_second_class
        self.x = x
        self.titles = titles
        self.fig_title = fig_title
        self.tick_points = 13

    def plot_main(self, title, legend=""):
        fig, ax = plt.subplots()
        ax.set_title(title)
        ticks_label = []
        logspac= np.logspace(-2,2,self.tick_points)
        for i in range(0,self.tick_points):
            ticks_label.append(str(logspac[i])[0:6])
        ticks = np.linspace(0, 100, self.tick_points)
        plt.xticks(ticks=ticks, labels=ticks_label)
        ax.plot(self.x, self.accuracy_main_class, linewidth=2.0)
        #ax.set_xscale('log')
        plt.show()

    def plot_second(self, title, legend=""):
        fig, ax = plt.subplots()
        ax.set_title(title)
        ticks_label = []
        logspac= np.logspace(-2,2,self.tick_points)
        for i in range(0,self.tick_points):
            ticks_label.append(str(logspac[i])[0:6])
        ticks = np.linspace(0, 100, self.tick_points)
        plt.xticks(ticks=ticks, labels=ticks_label)
        ax.plot(self.x, self.accuracy_second_class, linewidth=2.0)
        #ax.set_xscale('log')
        plt.show()
    
    def plot_3d(self, legend=""):
        ax = plt.axes(projection='3d')
        ticks_label = []
        
        logspac= np.logspace(-2,2,self.tick_points)
        for i in range(0,self.tick_points):
            ticks_label.append(str(logspac[i])[0:6])
        ticks = np.linspace(0, 100, self.tick_points)
        plt.xticks(ticks=ticks, labels=ticks_label)
        ax.set_title('Plot 3D',
             fontsize = 14)
        X, Y, Z = self.x, self.accuracy_main_class, self.accuracy_second_class
        ax.plot3D(X , Y, Z, 'gray')
        '''
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
        '''
        #ax.set_xscale('log')
        plt.show()

    def plot_metric(self, title, legend=""):
        fig, ax = plt.subplots()
        ax.set_title(title)
        ticks_label = []        
        logspac= np.logspace(-2,2,self.tick_points)
        for i in range(0,self.tick_points):
            ticks_label.append(str(logspac[i])[0:6])
        ticks = np.linspace(0, 100, self.tick_points)
        plt.xticks(ticks=ticks, labels=ticks_label)
        y = []
        for i in range(0,len(self.accuracy_main_class)):
            k = (self.accuracy_main_class[i]+self.accuracy_second_class[i])/2
            y.append(k)
        
        ax.plot(self.x, y, linewidth=2.0)
        #ax.set_xscale('log')
        plt.show()

    def plot_single(self, titles):
        self.plot_main(titles['main'])
        self.plot_second(titles['second'])
        self.plot_metric(titles['metric'])

    def plot_together(self):
        fig = plt.figure(figsize=plt.figaspect(5.))
        fig.suptitle(self.fig_title, fontsize=16)
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        #ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax4 = fig.add_subplot(2, 2, 3)
        X, Y, Z = self.x, self.accuracy_main_class, self.accuracy_second_class
        y = []
        ticks_label = []
        for i in range(0,len(self.accuracy_main_class)):
            k = (self.accuracy_main_class[i]+self.accuracy_second_class[i])/2
            y.append(k)
        logspac= np.logspace(-2,2,25)
        for i in range(0,25):
            ticks_label.append(str(logspac[i])[0:6])
        ticks = np.linspace(0, 100, 25)
        plt.xticks(ticks=ticks, labels=ticks_label)
        ax1.set_title(self.titles[0])
        ax2.set_title(self.titles[1])
        #ax3.set_title(self.titles[2])
        ax4.set_title(self.titles[3])
        ax1.plot(self.x, self.accuracy_main_class, linewidth=2.0)
        ax2.plot(self.x, self.accuracy_second_class, linewidth=2.0)
        #ax3.plot(X,Y,Z)
        ax4.plot(self.x, y, linewidth=2.0)



class Histograms:
    def __init__(self, labels, percentages, y_labels, title):
        self.labels = labels
        self.percentages = percentages
        self.y_labels = y_labels
        self.title = title

    def plot(self):
        fig, ax = plt.subplots()
        bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
        ax.bar(self.labels, self.percentages, label=self.labels, color=bar_colors[:len(self.labels)-1])
        ax.set_ylabel(self.y_labels)
        ax.set_title(self.title)
        #plt.yticks(self.percentages)
        plt.yticks(np.arange(40,80,5 ))
        plt.ylim(40,80)
        plt.grid()
        plt.show()



