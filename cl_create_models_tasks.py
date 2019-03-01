#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:04:33 2019

@author: ivan
"""
import os
import yaml, collections
import sys
import numpy as np


class PerturbedModelsTasks(object):

    def __init__(self):
        # for working with yaml files
        _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
        yaml.add_representer(collections.OrderedDict, self.dict_representer)
        yaml.add_constructor(_mapping_tag, self.dict_constructor)

        self.model_paths = (
            '/home/ivan/work/Project/Software/grl/src/grl/addons/rbdl/cfg/leo_vc',
            '/grl/src/grl/addons/rbdl/cfg/leo_vc',
            )
        self.tm_noise = (np.arange(-3, +4) * 0.2).tolist()
        self.jf_noise = [0.0]


    def generate(self):
        models, names = self.create_models(self.model_paths, ['tm', 'jf'],
                                           {'tm_noise':self.tm_noise, 'jf_noise':self.jf_noise})
        tasks, names = self.create_tasks(models, names)
        return (tasks, names)


    def get_task(self, task, tm = None, jf = None):
        filename = 'cfg/perturbed/leo_' + task
        file_extension = '.yaml'
        if jf == None:
            foutname = '{}_tm_{:.03f}{}'.format(filename, tm, file_extension)
        elif tm == None:
            foutname = '{}_jf_{:.03f}{}'.format(filename, jf, file_extension)
        else:
            foutname = '{}_tm_{:.03f}_jf_{:.03f}{}'.format(filename, tm, jf, file_extension)
        assert(os.path.exists(foutname))
        return foutname


    def create_models(self, paths, options, noise, join=True):
        for path in paths:
            if os.path.isdir(path):
                break

        ppath = '/~perturbed~'
        if not os.path.exists(path+ppath):
            os.makedirs(path+ppath)

        files = {
                'tf': '{}{}/leo_ff_dl{}_tf.lua',
                'no': '{}{}/leo_ff_dl{}.lua',
                }

        torsoMass = 0.94226
        torsoMassPro = noise['tm_noise']
        jointFriction = noise['jf_noise']

        content = {}
        for key in files:
            with open(files[key].format(path, '', ''), 'r') as content_file:
                content[key] = content_file.read()

        models = []
        names = []

        if not join:
            if 'tm' in options:
                for tmp in torsoMassPro:
                    model = {}
                    for key in content:
                        filename, file_extension = os.path.splitext(files[key].format(path,ppath,'_perturbed'))
                        foutname = '{}_tm_{:.03f}{}'.format(filename, tmp, file_extension)
                        with open(foutname, 'w') as fout:
                            new_mass = 'torsoMass = {}'.format(torsoMass*(1+tmp))
                            fout.write( content[key].replace('torsoMass = 0.94226', new_mass) )
                        if key == 'tf':
                            model['balancing_tf'] = foutname
                        else:
                            model['balancing'] = model['walking'] = foutname
                    models.append(model)
                    names.append('tm_{:.03f}'.format(tmp))

            if 'jf' in options:
                for jfp in jointFriction:
                    model = {}
                    for key in content:
                        filename, file_extension = os.path.splitext(files[key].format(path,ppath,'_perturbed'))
                        foutname = '{}_jf_{:.03f}{}'.format(filename, jfp, file_extension)
                        with open(foutname, 'w') as fout:
                            new_friction = 'jointFriction = {}'.format(jfp)
                            fout.write( content[key].replace('jointFriction = 0.00', new_friction) )
                        if key == 'tf':
                            model['balancing_tf'] = foutname
                        else:
                            model['balancing'] = model['walking'] = foutname
                    models.append(model)
                    names.append('jf_{:.03f}'.format(jfp))
        else:
            if 'tm' in options and 'jf' in options:
                for tmp in torsoMassPro:
                    for jfp in jointFriction:
                        model = {}
                        for key in content:
                            filename, file_extension = os.path.splitext(files[key].format(path,ppath,'_perturbed'))
                            foutname = '{}_tm_{:.03f}_jf_{:.03f}{}'.format(filename, tmp, jfp, file_extension)
                            with open(foutname, 'w') as fout:
                                new_mass = 'torsoMass = {}'.format(torsoMass*(1+tmp))
                                new_content = content[key].replace('torsoMass = 0.94226', new_mass)
                                new_friction = 'jointFriction = {}'.format(jfp)
                                new_content = new_content.replace('jointFriction = 0.00', new_friction)
                                fout.write(new_content)
                            if key == 'tf':
                                model['balancing_tf'] = foutname
                            else:
                                model['balancing'] = model['walking'] = foutname
                        models.append(model)
                        names.append('tm_{:.03f}_jf_{:.03f}'.format(tmp, jfp))

        return models, names


    def create_tasks(self, models, names):

        if not os.path.exists('cfg/perturbed/'):
            os.makedirs('cfg/perturbed/')

        itasks = {
            'balancing_tf': 'cfg/leo_balancing_tf.yaml',
            'balancing':    'cfg/leo_balancing.yaml',
            'walking':      'cfg/leo_walking.yaml'
            }

        otasks = []
        for model, name in zip(models,names):
            task = {}
            for key in itasks:
                conf = self.read_cfg(itasks[key])
                conf['environment']['environment']['model']['dynamics']['file'] = model[key]
                path, filename = os.path.split(itasks[key])
                filename, file_extension = os.path.splitext(filename)
                fullname = path + '/perturbed/' + filename + '_' + name + file_extension
                self.write_cfg(fullname, conf)
                task[key] = fullname
            otasks.append(task)

        return otasks, names


    def read_cfg(self, cfg):
        """Read configuration file"""
        # check if file exists
        yfile = cfg
        if os.path.isfile(yfile) == False:
            print('File %s not found' % yfile)
            sys.exit()

        # open configuration
        stream = open(yfile, 'r')
        conf = yaml.load(stream)
        stream.close()
        return conf


    def write_cfg(self, outCfg, conf):
        """Write configuration file"""
        # create local yaml configuration file
        outfile = open(outCfg, 'w')
        yaml.dump(conf, outfile)
        outfile.close()


    def dict_representer(self, dumper, data):
      return dumper.represent_dict(data.items())


    def dict_constructor(self, loader, node):
      return collections.OrderedDict(loader.construct_pairs(node))
