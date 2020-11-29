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
from mgedt import mgedt_gui
from threading import Thread
import os
import webbrowser

UPDATE_RATE=1
global PARAMS_SET
PARAMS_SET=False
global PARAMS_USER
PARAMS_USER = []

app = Flask(__name__, template_folder='web')
thread = Thread(target = mgedt_gui)

def update_interace_info():
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
        
        time.sleep(UPDATE_RATE)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prog')
def prog():
    return render_template('prog.html')

@app.route('/start', methods=['POST', 'GET'])
def start():
    thread.start()
    return redirect("prog")

@app.route('/progress')
def progress():
    return Response(update_interace_info(), mimetype= 'text/event-stream')

@app.route('/parameters', methods=['POST', 'GET'])
def parameters():
    if request.method == 'POST':
        args = parse_args(request.form)
        from algorithm.parameters import set_params
        set_params(args)
        global PARAMS_SET
        PARAMS_SET=True
        global PARAMS_USER
        PARAMS_USER=args
        return redirect("/")
    else:
        return render_template('parameters.html')

@app.route('/check_params')
def check_params():
    return Response(check_params_aux(), mimetype= 'text/event-stream')

def check_params_aux():
    global PARAMS_SET
    ret_string = "data:" + json.dumps(PARAMS_SET) + "\n\n"
    yield ret_string

def get_default_params():
    from UI_parameters import schema
    ret_string = "data:" + json.dumps(schema) + "\n\n"
    yield ret_string

def start_app():
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new("http://127.0.0.1:5000/")
    app.run(debug=False)

def parse_args(args):
    list_args = []
    for key, val in args.items():
        if val == "True":
            list_args.append("--"+key)
        elif val == "False" or val=="":
            continue
        else:
            list_args.append("--"+key+"="+val)
    return list_args



if __name__ == "__main__":
    start_app()