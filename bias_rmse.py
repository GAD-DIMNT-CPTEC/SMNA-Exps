#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import dask
from dask.diagnostics import ProgressBar

ProgressBar().register()
dask.config.set(scheduler="threads")

paths = {
    "EXP1": "/mnt/beegfs/carlos.bastarz/SMNA_v3.0.x_check/anls_compare/pos/convert_to_netcdf/output/zarr/EXP1.zarr",
    "EXP2": "/mnt/beegfs/carlos.bastarz/SMNA_v3.0.x_check/anls_compare/pos/convert_to_netcdf/output/zarr/EXP2.zarr",
    "EXP3": "/mnt/beegfs/carlos.bastarz/SMNA_v3.0.x_check/anls_compare/pos/convert_to_netcdf/output/zarr/EXP3.zarr",
    "EXP4": "/mnt/beegfs/carlos.bastarz/SMNA_v3.0.x_check/anls_compare/pos/convert_to_netcdf/output/zarr/EXP4.zarr",
    "EXP5": "/mnt/beegfs/carlos.bastarz/SMNA_v3.0.x_check/anls_compare/pos/convert_to_netcdf/output/zarr/EXP5.zarr",
    "EXP6": "/mnt/beegfs/carlos.bastarz/SMNA_v3.0.x_check/anls_compare/pos/convert_to_netcdf/output/zarr/EXP6.zarr",
    "EXP7": "/mnt/beegfs/carlos.bastarz/SMNA_v3.0.x_check/anls_compare/pos/convert_to_netcdf/output/zarr/EXP7.zarr",
}

datasets = [xr.open_zarr(p, chunks={}) for p in paths.values()]

ds = xr.combine_nested(
    datasets,
    concat_dim="exp",
    coords="minimal",
    compat="override"
)

ds = ds.assign_coords(exp=list(paths.keys()))

ds = ds.chunk({
    "exp": 1,
    "cycle": 1,
    "lead": 5,
    "lat": 90,
    "lon": 180
})

weights = np.cos(np.deg2rad(ds.lat))
weights.name = "weights"

def errorf(var):
    field = ds[var]
    analysis = field.sel(lead=0)
    error = field - analysis
    return error

def scores(var):
    error = errorf(var)
    bias_lead = error.weighted(weights).mean(("lat","lon","lead"))
    rmse_lead = np.sqrt((error**2).weighted(weights).mean(("lat","lon","lead")))
    bias_cycle = error.weighted(weights).mean(("lat","lon","cycle"))
    rmse_cycle = np.sqrt((error**2).weighted(weights).mean(("lat","lon","cycle")))
    return bias_lead, rmse_lead, bias_cycle, rmse_cycle     

Vars = ['pslc', 'psnm', 'uvel', 'vvel', 'temp', 'umes', 'zgeo', 'agpl', 'tp2m', 'u10m', 'v10m', 'q02m']

for var in Vars:
    print(var)
    bias_lead, rmse_lead, bias_cycle, rmse_cycle = scores(var)
    bias_lead, rmse_lead, bias_cycle, rmse_cycle = dask.compute(bias_lead, rmse_lead, bias_cycle, rmse_cycle)

    EXPS_SCORES = xr.Dataset({
        "bias_lead": bias_lead,
        "rmse_lead": rmse_lead,
        "bias_cycle": bias_cycle,
        "rmse_cycle": rmse_cycle
    })

    EXPS_SCORES.to_zarr("zarr/EXPS_SCORES_" + str(var) + ".zarr", mode="w")
