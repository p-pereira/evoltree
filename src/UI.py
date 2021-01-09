# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 22:35:09 2020

@author: pedro
"""
from flask import Flask, render_template, Response, redirect, request
from math import log
import time
import json
import numpy as np
from mgedt import MGEDT
#from mgedt import mgedt_gui
from threading import Thread
import os
import webbrowser


#global mgedt_gui
#mgedt_gui = None
app = Flask(__name__, template_folder='web')
#thread = Thread(target = mgedt_gui.evolve)

class MGEDT_UI:
    def __init__(self, update_rate=1):
        self.UPDATE_RATE = update_rate
        self.PARAMS_SET = False
        self._mgedt = None
        self.thread = None

    def update_interace_info(self):
        from stats.stats import stats
        from utilities.stats import trackers
        from algorithm.parameters import params
        generation = 0
        total_gens = params["GENERATIONS"]
        vid_dict = {}
        
        while generation <= total_gens:
            
            vid_dict['total'] =  total_gens
            try :
                generation = stats["gen"]
                vid_dict['gen'] = generation
                percentage = round(min((generation * 100) / total_gens, 100),2)
                vid_dict['percentage'] = percentage
            except Exception as e:
                vid_dict['gen'] = 0
                vid_dict['percentage'] = 0
                print(e)
            vid_dict['label'] = generation
            if generation > 0:
                g = generation - 1
                best_auc = trackers.best_fitness_list[g][0]*-1
                mean_auc = np.mean(trackers.first_pareto_list[g][0])*-1
                data = [best_auc, mean_auc]
                
                pareto_auc = [0.5 * round(x/0.5) for x in trackers.first_pareto_list[g][0]]
                pareto_comp = [log(y,10) for y in trackers.first_pareto_list[g][1]]
                pareto = {'data' : [{'x': x, 'y': y} for x,y in zip(pareto_auc,
                                                                    pareto_comp)],
                          'fill' : False
                          }
            else:
                data = []
                pareto_auc = []
                pareto = [{}]
            
            vid_dict['data_graphic'] = data
            vid_dict['pareto_auc'] = pareto_auc
            vid_dict['pareto'] = pareto
            ret_string = "data:" + json.dumps(vid_dict) + "\n\n"
            yield ret_string
            
            time.sleep(self.UPDATE_RATE)

    def check_params_aux(self):
        ret_string = "data:" + json.dumps(self.PARAMS_SET) + "\n\n"
        yield ret_string
    
    def start_app(self):
        if not os.environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new("http://127.0.0.1:5000/")
        app.run(debug=False)
    
    def parse_args(self, args):
        list_args = []
        for key, val in args.items():
            if val == "True":
                list_args.append("--"+key)
            elif val == "False" or val=="":
                continue
            else:
                list_args.append("--"+key+"="+val)
        return list_args

    def start_pop(self, args):
        self._mgedt = MGEDT(UI_params=args.to_dict(), UI=True)
        self.PARAMS_SET = True
        self.thread = Thread(target = self._mgedt.fit)
        

@app.route('/')
def index():
    global mgedt_gui
    if mgedt_gui==None:
        mgedt_gui = MGEDT_UI()
    return render_template('index.html')

@app.route('/prog')
def prog():
    return render_template('prog.html')

@app.route('/start', methods=['POST', 'GET'])
def start():
    global mgedt_gui
    mgedt_gui.thread.start()
    return redirect("prog")

@app.route('/progress')
def progress():
    return Response(mgedt_gui.update_interace_info(), mimetype= 'text/event-stream')

@app.route('/parameters', methods=['POST', 'GET'])
def parameters():
    if request.method == 'POST':
        args = request.form
        mgedt_gui.start_pop(args)
        return redirect("/")
    else:
        return render_template('parameters.html')

@app.route('/check_params')
def check_params():
    return Response(mgedt_gui.check_params_aux(), mimetype= 'text/event-stream')





if __name__ == "__main__":
    global mgedt_gui
    
    mgedt_gui = MGEDT_UI()
    mgedt_gui.start_app()