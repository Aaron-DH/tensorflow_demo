#!/usr/bin/python

import os
import json

def parse_cluster_spec():
    result = {}
    cluster_spec = os.environ.get("CLUSTER_SPEC")
    job_strings = cluster_spec.split(",")
    if not cluster_spec:
        raise ValueError("Empty cluster_spec string")
    for job_string in job_strings:
        if job_string.count("|") != 1:
          raise ValueError("Not exactly one instance of '|' in cluster_spec")
        job_name = job_string.split("|")[0]
        job_tasks = job_string.split("|")[1].replace(";", ",")
        if job_name == "ps":
            #print ("ps_hosts:" + str(job_tasks))
            result.update({"ps_hosts":job_tasks})
        else:
            #print ("worker_hosts:" + str(job_tasks))
            result.update({"worker_hosts":job_tasks})
    #hostname = os.environ.get("HOSTNAME")
    #print ("job_name:" + hostname.split("-")[-2])
    #print ("task_index:" + hostname.split("-")[-1])
    #result.update(
    #{
    #    "job_name":os.environ.get("RESOURCE_NAME"),
    #    "task_index":hostname.split("-")[-1]
    #})
    return result

if __name__ == "__main__":
    result = parse_cluster_spec()
    print (json.dumps(result))