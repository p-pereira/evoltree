# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 22:35:09 2020

@author: pedro
"""
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8

from flask import Flask, render_template, Response, request, url_for, redirect

from math import log
import time
import json

app = Flask(__name__, template_folder='web')

"""
@app.route("/<int:bars_count>/")
def chart(bars_count):
    if bars_count <= 0:
        bars_count = 1
    return render_template("main.html", bars_count=bars_count)

@app.route('/bokeh')
def bokeh():

    # init a basic bar chart:
    # http://bokeh.pydata.org/en/latest/docs/user_guide/plotting.html#bars
    fig = figure(plot_width=600, plot_height=600)
    fig.vbar(
        x=[1, 2, 3, 4],
        width=0.5,
        bottom=0,
        top=[1.7, 2.2, 4.6, 3.9],
        color='navy'
    )

    # grab the static resources
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    # render template
    script, div = components(fig)
    html = render_template(
        'graphic.html',
        plot_script=script,
        plot_div=div,
        js_resources=js_resources,
        css_resources=css_resources,
    )
    return encode_utf8(html)
"""

from multiprocessing import Pool
from algorithm.parameters import params, set_params
from fitness.evaluation import evaluate_fitness
from stats.stats import stats, get_stats
from operators.initialisation import initialisation
from utilities.algorithm.initialise_run import pool_init
import json
from os import path
import pandas as pd
import numpy as np

def mgedt_gui():
    if params['MULTICORE']:
        # initialize pool once, if mutlicore is enabled
        params['POOL'] = Pool(processes=params['CORES'], initializer=pool_init,
                              initargs=(params,))  # , maxtasksperchild=1)

    # Initialise population
    individuals = initialisation(params['POPULATION_SIZE'])
    
    # Evaluate initial population
    individuals = evaluate_fitness(individuals)
    
    if params['SAVE_POP']:
        filename1 = path.join(params['FILE_PATH'], 'Begin-initialPop.txt')
        with open(filename1, 'w+', encoding="utf-8") as f:
            for item in individuals:
                f.write("%s\n" % item)
            f.close()
    
    # Generate statistics for run so far
    get_stats(individuals)
    
    total_gens = params['GENERATIONS']+1
    # Traditional GE
    for generation in range(1, total_gens):
        # GUI
        #vid_dict = {}
        #vid_dict[0] = min((generation * 100) / total_gens, 100)
        #yield "data:" + str(x) + "\n\n"
        #ret_string = "data:" + json.dumps(vid_dict) + "\n\n"
        #print(ret_string)
        #yield ret_string
        stats['gen'] = generation
        # New generation
        individuals = params['STEP'](individuals)
        
    
    get_stats(individuals, end=True)

from threading import Thread
thread = Thread(target = mgedt_gui)
#-------------------------------
# Configuration of the application.
# num_bars: number of progress bars to render
# prog_inc: how mcuh the progress bar increases per update
# update_rate: how frequently to update the progress bar, in seconds
#-------------------------------
class Config:
    num_bars = 1
    prog_inc = 20
    update_rate = 1

# Instantiate app_config

app_cfg = Config
def generate():
    generation = 0
    total_gens = params["GENERATIONS"]
    vid_dict = {}
    
    while generation <= total_gens:
        from stats.stats import stats
        from utilities.stats import trackers
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
            
            pareto_auc = [round(x) for x in trackers.first_pareto_list[g][0]]
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
        
        time.sleep(app_cfg.update_rate)
        

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prog')
def prog():
    return render_template('prog.html', num_bars = app_cfg.num_bars)

@app.route('/start', methods=['POST', 'GET'])
def start():
    ## create a new row in GitTasks table, and use its PK(id) as task_id
    #task_id = create_new_task_row()
    set_params('')
    thread.start()
    return redirect("prog")

@app.route('/progress')
def progress():
    return Response(generate(), mimetype= 'text/event-stream')

#app.add_url_rule('/', 'prog', mgedt_gui)

if __name__ == "__main__":
    app.run(debug=True)