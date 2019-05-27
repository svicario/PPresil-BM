from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from linear_model import BayesianLinearModel, _model_evidence, _negative_log_marginal_likelihood
from patsy.contrasts import Sum, Treatment
import copy
import numpy as np
import pandas as pd
import xarray


def LoadXarray(path, maxvalue=10000,navalue=-32768, bandNameFormatter=pd.to_datetime):
    TDG=xarray.open_rasterio(path)
    TDG.coords["band"]=bandNameFormatter([x.strip() for x in TDG.attrs['band_names'].split(",")])
    if TDG.dtype.name.find("int")>-1:
        attrs=TDG.attrs
        TDG=TDG.where(TDG!=navalue)
        TDG=TDG/maxvalue
        TDG.attrs=attrs
    if TDG.attrs["coordinate_system_string"].find("UTM")!=-1:
        TDG.coords["x"].attrs["units"]=TDG.coords["y"].attrs["units"]="meters"
    TDG.coords["x"].attrs["long_name"]="Longitude"
    TDG.coords["y"].attrs["long_name"]="Latitude"
    return TDG

def MakeLinearSinus(t, k=[1,2,3], trend=False, YAM=False, freq=365):
    """
    Actual Harmonic model expressed as  transformation of time variable necessary 
    to fit the data within a linear framework
    """
    N=len(t)
    K=np.repeat(np.array([k]),N, axis=0)
    fix=2*np.pi/freq
    Fix=t.reshape(N,1)*fix*K
    #print(Fix.shape)
    names=[]
    if trend:
        Xm=np.concatenate([np.array([1]*N).reshape(N,1),
                           t.reshape(N,1),np.sin(Fix), np.cos(Fix)],axis=1)
        names=["intercept","trend"]+["sin"+str(k) for k in K ]+["cos"+str(k) for k in K ]
    elif YAM:
        year=((t/365).astype("int")).flatten()
        #year = Sum().code_without_intercept(list(set(year))).matrix[year,:]
        #print(Treatment().code_without_intercept(list(set(year))))
        #To deal with year that have no observation make an index that guide observation to correct contrast
        ex=Treatment().code_without_intercept(list(set(year)))
        #print(year[0])
        #print(ex.matrix.shape)
        cast=pd.Series(np.arange(ex.matrix.shape[0]), index=[0]+[int(x.split(".")[1][:-1]) for x in ex.column_suffixes])
        #print(cast)
        #inc=cast.loc[year]
        year = ex.matrix[cast.loc[year],:]
        #print(year.shape[1])
        #year=sm.tools.categorical(year, drop=True)
        Xm=np.concatenate([np.array([1]*N).reshape(N,1),
                           year,np.sin(Fix), np.cos(Fix)],axis=1)
        names=["intercept"]+["year"+str(k) for k in 2+np.arange(year.shape[1]) ]+["sin"+str(kk) for kk in k ]+["cos"+str(kk) for kk in k ]
    else:
        Xm=np.concatenate([np.array([1]*N).reshape(N,1),
                           np.sin(Fix), np.cos(Fix)],axis=1)
        names=["intercept"]+["sin"+str(kk) for kk in k ]+["cos"+str(kk) for kk in k ]
    return Xm,names

def RunnerBayesian(values,times, change=True,PP=False, name=False, k=[1,2,3],
                   fulloutput=False,reps=100, Event=False, BFT=1.6,freqsubsample=15,reformat_dict={}):
    """
    Run a pixel columns and get output
    """
    freq=365
    if reformat_dict:
        values=values.astype("float")
        values[reformat_dict["nan"]==values]=np.nan
        values=values/reformat_dict["maxvalues"]
    fullyear=np.ceil(times[-1]/freq).astype("int")
    times=times[np.isfinite(values)]
    values=values[np.isfinite(values)]
    #delta=(times[0]-times[-1])
    #if int(delta/freq)==int(delta/freq):
    #    times=times[:-1]
    #    values=values[:-1]
    if change:
        Model=MultiModel
    else:
        Model=MonoModel
    try:
        models, evidences, check=Model(times,values,freq=freq, k=k,BFT=BFT)
    except:
        #to handle nan-only columns
        return np.array([np.nan]*(fullyear*3*2+2))
    modelA=models
    Res, Rest=FeatureBayes2(modelA, obsdays=times, obsvalue=values, freq=freq,PP=PP, Event=Event,reps=reps,freqsubsample=freqsubsample, YearsL=fullyear)
    R=Res.stack()
    R[R==-1]=np.nan
    if PP:
        RT=Rest.stack()
        R=np.concatenate([R,RT])
    if fulloutput:
        return models, Res, Rest 
    if name:
        name=["_".join(x) for x in R.index]
        if PP:
            name+=["_".join(x) for x in RT.index]
        return R, name
    else:
        return R

def PartialModelEvidence2(model, T,y):
    """
    Calculate the increase in evidence adding new data, without modifying original model
    """
    Imodel=copy.deepcopy(model)
    Imodel.update(T[:,np.newaxis],y[:,np.newaxis])
    base=0
    if model._BayesianLinearModel__initialised:
        base=model.evidence()
    return Imodel.evidence()-base
def MonoModel(t,YY,freq,k=[1,2,3], BFT=1.6):
    HarmonicA = FunctionTransformer(lambda x: MakeLinearSinus(x, k=k,YAM=True)[0], validate=False)
    modelA=BayesianLinearModel(basis=HarmonicA.fit_transform)
    modelA.update(t[:,np.newaxis],YY[:,np.newaxis])
    return modelA, [],[]
def MultiModel(t,YY,freq=365, k=[1,2,3],BFT=1.6):
    """
    Fit a model year by year switching between the previous year  model, a new Seasonal model and a  new linear model
    """
    HarmonicA = FunctionTransformer(lambda x: MakeLinearSinus(x, k=k,YAM=True)[0], validate=False)
    modelA=BayesianLinearModel(basis=HarmonicA.fit_transform)
    nameparam=MakeLinearSinus(t[:,np.newaxis], k=k,YAM=True, freq=365)[1]
    noyearparam=[x for x,y in enumerate(nameparam) if y.find("year")!=0]
    modelA.update(t[:,np.newaxis],YY[:,np.newaxis])
    Harmonic = FunctionTransformer(lambda x: MakeLinearSinus(x, k=k,YAM=False)[0], validate=False)
    g=len(t)
    startingPoint={"location":modelA.location[noyearparam],
                    "dispersion":modelA.dispersion[noyearparam,:][:,noyearparam]*(g**0.5),
                    "scale":modelA.scale,
                   "shape":modelA.shape}
    
    #     startingPoint={"location":modelA.location,
    #                 "dispersion":modelA.dispersion*g,
    #                 "scale":modelA.scale,
    #                "shape":modelA.shape}
    model=BayesianLinearModel(basis=Harmonic.fit_transform, **startingPoint)
    yearStart=0
    #therdshold for bayesian factor, assuming natural log of the ratio
    #BFT=4
    evidences=[np.array([-np.inf])]
    evits=[]
    eviss=[]
    evifs=[]
    maxvalue=YY.max()
    cont=[]
    models=[]
    import copy
    for year in np.arange(yearStart,np.ceil(t.max()/freq)):
    #for year in np.arange(3):
        T=t[(t>(freq*year))&(t<(freq*(year+1)))]
        y=YY[(t>(freq*year))&(t<(freq*(year+1)))]
        evit=PartialModelEvidence2(model, T,y)
        evits.append(evit)
        #New flat model
        evif=np.nan
        if len(T)>3:
            modelF=BayesianLinearModel(basis=lambda x:np.concatenate([np.array([1]*len(x))[:,np.newaxis],x], axis=1))
            modelF.update(T[:,np.newaxis],y[:,np.newaxis])
            if modelF.location[0]<0.2:
                evif=modelF.evidence()
        evifs.append(evif)
        evis=np.nan
        #NewHarmonic model
        if len(T)>2:
            #modelS=BayesianLinearModel(basis=Harmonic.fit_transform)
            #startingPoint={}
            modelS=BayesianLinearModel(basis=Harmonic.fit_transform, **startingPoint)
        
            modelS.update(T[:,np.newaxis],y[:,np.newaxis])
            evis=modelS.evidence()
        eviss.append(evis)
        #Decision
        if year+1984==2012:
            pass
            #print(evis, evit)
        if evis-evit>BFT:
            model=modelS
            cont.append("new")

        elif evif-evit>BFT:
            model=modelF
            cont.append("flat")
        else:
            model.update(T[:,np.newaxis],y[:,np.newaxis])
            #print(model.location)
            cont.append("old")

        evi=model.evidence()
        models.append(model)    
        evidences.append(np.array([evi]))
    evidences=[np.concatenate(evidences, axis=0)]
    evidences=evidences[0][1:]
    return ModelList(models), evidences, (eviss,evits, evifs, cont)

def FeatureBayes2(modelA, obsdays, obsvalue, freq=365,PP=True, Event=False,reps=500,freqsubsample=15, YearsL=None):
    """
    Extract 4 descriptors of the time series from a model or multiyear model and perfrom a PP test of goodness of fit 
    """
    t=obsdays
    YY=obsvalue
    Years=np.ceil(t.max()/freq).astype("int")
    freqsubsample=freqsubsample
    #daystoobs=np.arange(Years*freq)
    daystoobs=np.arange(0,Years*freq, freqsubsample)
    X=modelA._BayesianLinearModel__basis(daystoobs)
    #pred0=pd.DataFrame(np.array([np.nansum(X*Randomizer(modelA,daystoobs),axis=1) for x in np.arange(reps)]))
    #params=modelA.random(samples=1000)
    #print(X.shape,Years,freqsubsample)
    try:
        params=Randomizer(modelA,daystoobs,reps)
    except ValueError:
        print(obsvalue)
    pred0=pd.DataFrame(np.nansum(X[:,np.newaxis,:]*params,axis=2).T)
    Res=[]
    Rest=[]
    varyear=0
    totdays=pred0.shape[1]
    if PP or Event:
        X=modelA._BayesianLinearModel__basis(t)
        #pred0t=pd.DataFrame(np.array([np.nansum(X*Randomizer(modelA,t),axis=1) for x in np.arange(reps)]))
        paramst=Randomizer(modelA,t,reps)
        pred0t=pd.DataFrame(np.nansum(X[:,np.newaxis,:]*paramst,axis=2).T)
        #pred0t=pred0.iloc[:,t-1]
        yeart=(t/freq).astype("int")
    yearobs=(daystoobs/freq).astype("int")
    for year in np.arange(Years):
        res=[]
        
        #f=pred0.iloc[:,year*freq:(year+1)*freq].values
        f=pred0.iloc[:,yearobs==year].values
        daystoobsY=daystoobs[yearobs==year]
        if PP or Event:
            ft=pred0t.iloc[:,yeart==year].values
            ftt=YY[yeart==year]
            if PP:
                M=ft.mean(axis=1)
                Mm=M.mean()
                STD=ft.std(axis=1)
                STDm=STD.mean()
                argMAX=ft.argmax(axis=1)
                argMAXm=argMAX.mean()
                rest=((np.abs(M-Mm)>=np.abs(ftt.mean()-Mm)).sum(),
                      (np.abs(STD-STDm)>=np.abs(ftt.std()-STDm)).sum(),(np.abs(argMAX-argMAXm)>=np.abs(ftt.argmax()-argMAXm)).sum())
                Rest.append(rest)
            if Event:
                #in caso di due sole osservazioni in un anno meglio saltare
                if (yeart==year).sum()< 3: 
                    res.append(pd.Series([np.nan,np.nan], name="event", index=["mean","sd"]))
                    continue
                tt=t[yeart==year]
                stat=np.diff(ft,axis=1)/np.diff(tt)
                try:
                    temp=(-stat).argmax(axis=1)
                except ValueError:
                    print(stat)
                    print(ft)
                    print(t,YY)
                #print(daystoobsY.shape,temp.max())
                eventf=tt[temp]
                res.append(pd.Series([eventf.mean()-(year*freq),eventf.std()], name="event", index=["mean","sd"]))
            
        
        ff=f.mean(axis=1)
        res.append(pd.Series([ff.mean(),ff.std()], name="mean", index=["mean","sd"]))
        ff=f.std(axis=1)
        varyear+=(ff**2)*f.shape[1]/totdays
        res.append(pd.Series([ff.mean(),ff.std()], name="stdintra", index=["mean","sd"]))
        #res.append(pd.Series([f.idxmax(axis=1).mean(),f.idxmax(axis=1).std()], name="maxpos", index=["mean","sd"]))
        ff=daystoobsY[f.argmax(axis=1)]
        ffmaxpos=ff-year*freq
        res.append(pd.Series([ffmaxpos.mean(),ffmaxpos.std()], name="maxpos", index=["mean","sd"]))
        #res.append(pd.Series([ff.mean(),ff.std()], name="maxpos", index=["mean","sd"]))
        Res.append(pd.DataFrame(res).stack())
        
    Res=pd.concat(Res, keys=np.arange(Years).astype("U2"), axis=1)
    varinter=(np.var(pred0.values,axis=1)-varyear)**0.5
    #Res.loc[("sdinter","mean"),str(Years)]=varinter.mean()
    #Res.loc[("sdinter","sd"),str(Years)]=varinter.std()
    Res.loc[("sdinter","mean"),"all"]=varinter.mean()
    Res.loc[("sdinter","sd"),"all"]=varinter.std()
    for y in np.arange(Years+1,YearsL+1):
        Res.loc[:,str(y)]=-1
        Res.loc["sdinter",str(y)]=np.nan
    if PP:
        Rest=pd.DataFrame(Rest,columns=["mean","stdintra","maxPos"], index=np.arange(Years).astype("U2"))
        Rest=Rest/reps
    return Res, Rest

class HarmBayMod(object):
    """
    Class that handle runs of the model
    """
    def __init__(self, xarrayD , reformat_dict={}):
        """
        initialize: it assume that first dimension is time, then latitude, finally longitude
        """
        self.PP=False
        self.Event=False
        self.variableLarge=["maxpos","event"]
        self.RawData=xarrayD
        self.reformat_dict=reformat_dict
        
        #Transform, maybe not relevant
        self.nt,self.ny, self.nx=self.RawData.shape
        self.transform=self.RawData.attrs["transform"]
        if len(self.transform)==9:
            self.transform=self.transform[:6]
        self.dx,mx,self.xmin,my,self.dy,self.ymax=self.transform
        self.xmin=self.RawData.x.values[0]-(self.dx/2)
        self.ymin=self.RawData.y.values[0]-(self.dy/2)
        self.transform=np.array([self.dx,mx,self.xmin,my,self.dy,self.ymax])
        assert (mx==my)&(mx==0), "no rotation expected"
        #CRS
        try:
            self.crs=self.RawData.attrs["coordinate_system_string"]
        except KeyError:
            self.crs=""
    def RunStackBayesian(self, memoryR=True, PP=False, Event=False, change=True, bandNameFormatter=pd.to_datetime):
        """
        Run over a cube
        """
        self.PP=PP
        self.Event=Event
        self.change=change
        xarrayD=self.RawData
        self.start=xarrayD[:,0,0].band[0].values.astype('datetime64[Y]')
        try:
            self.tff=(xarrayD[:,0,0].band-self.start).values.astype('timedelta64[D]').astype("int")
        except TypeError:
            time=bandNameFormatter([x.strip() for x in xarrayD.attrs['band_names'].split(",")])
            self.start=time[0:1].values[0].astype("datetime64[Y]")
            self.tff=(time-self.start).values.astype('timedelta64[D]').astype("int")
        #tff=(MSAVIOLD[:,0,0].band-np.array([startdate]).astype('datetime64[ns]')).values.astype('timedelta64[D]').astype("int")
        self.Data=np.apply_along_axis(arr=xarrayD.values, axis=0, func1d=RunnerBayesian, times=self.tff, change=change,PP=PP, Event=Event,reformat_dict=self.reformat_dict)
        if memoryR:
            self.MemoryRelease()
        return self.Data
    def MemoryRelease(self):
        """
        erase data to release memory after run to alleviate 
        """
        del self.RawData
    def FormatBayes(self, namefile):
        """
        Build a coherent netcdf Dataset  with multiple variables from the single array of output
        """
        tff=self.tff
        start=self.start.astype("int")+1970
        BDFBayes=self.Data
        BDFBayesx=xarray.DataArray(BDFBayes, dims=["Time","Y","X"])
        #BDFBayesx=BDFBayesx.rename(dim_0="Year",dim_1="y",dim_2="x")
        #produce random data to extract name
        Name=RunnerBayesian(values=np.random.random(len(self.tff)),times=self.tff, name=True, PP=self.PP, change=self.change,Event=self.Event)[1]
        #Build grid of affine trasform
        X=np.arange(self.nx)*self.dx+self.xmin+self.dx/2
        Y=np.arange(self.ny)*self.dy+self.ymax+self.dy/2
        #Annotate raw output
        BDFBayesx.coords["Y"]=(("Y"),Y)
        BDFBayesx.coords["X"]=(("X"),X)
        BDFBayesx.coords["X"].attrs["long_name"]="Longitude"
        BDFBayesx.coords["Y"].attrs["long_name"]="Latitude"
        #Year=[ int(x.split("_")[-1])+start for x in Name]
        Year=[ x.split("_")[-1] for x in Name]
        BDFBayesx.coords["Time"]=(("Time"),Year)
        ClassName=[ "_".join(x.split("_")[:-1]) for x in Name]
        BDFBayesx.coords["ClassName"]=(("Time"),ClassName)
        #Split raw output in the variables to build composite netcdf
        BDFBa=xarray.Dataset()
        for Class in [ x for x in BDFBayesx.ClassName if x not in ["sdinter_mean","sdinter_sd"]]:
            Class=str(Class.values)    
            BDFBa[Class]=BDFBayesx.where(BDFBayesx.ClassName==Class, drop=True)
            BDFBa[Class].attrs["grid_mapping"]="crs"
        del BDFBa.coords["ClassName"]
        #format year, once removed stat not related to time
        Year=[ int(x.split("_")[-1])+start for x in BDFBa.Year.values]
        BDFBa.coords["Year"]=(("Year"),Year)
        #Add variables with no time dimension
        BDFBa["sdinter_mean"]=(("y","x"),BDFBayesx[BDFBayesx.ClassName=="sdinter_mean",:,:][0,:,:])
        BDFBa["sdinter_mean"].attrs["grid_mapping"]="crs"
        BDFBa["sdinter_sd"]=(("y","x"),BDFBayesx[BDFBayesx.ClassName=="sdinter_sd",:,:][0,:,:])
        BDFBa["sdinter_sd"].attrs["grid_mapping"]="crs"
        BDFBa.coords["crs"]=0
        BDFBa.coords["crs"].attrs["crs_wkt"]=self.crs
        BDFBa.coords["crs"].attrs["spatial_ref"]=self.crs
        BDFBa.coords["crs"].attrs["transform"]=self.transform
        BDFBa.attrs["spatial_ref"]=self.crs
        BDFBa.attrs["transform"]=self.transform
        BDFBa.attrs["GeoTransform"]=self.transform
        #return BDFBa
        #encode and save
        enc={}
        enc.update([ (x,{'dtype': 'int16', 'scale_factor': BDFBa[x].max().round().astype("int").values/10000, '_FillValue': -9999}) for x in BDFBa.data_vars.keys()])
        #for var in self.variableLarge:
        #for var in ["maxpos","event"]:
        #    enc[var+'_mean']['scale_factor']=1
        #    enc[var+'_sd']['scale_factor']=1
        BDFBa.to_netcdf(namefile+".nc", encoding=enc)
        return BDFBa

class ModelList(object):
    """
    Class of multiyear model it simulate basic feature of Bayesian linear model class
    """
    def __init__(self, modellist=[], freq=365):
        self.freq=freq
        self.modellist=modellist
        self.maxparam=max([x.location.shape[0] for x in modellist]+[0])
    def predict(self,X, y=None, variance=False):
        """
        Calculate posterior predictive values.
        """
        res={}
        yeart=(X/self.freq).astype("int").flatten()
        for year, model in enumerate(self.modellist):
            out=model.predict(X[yeart==year], y, variance)
            if out.__class__==np.array([]).__class__:
                out=[out]
            for n,o in enumerate(out):
                res.setdefault(n,[]).append(o)
        #print(res)
        Res=[]
        axis=0
        for r in res:
            if (r==2): axis=1
            Res.append(np.concatenate(res[r],axis=axis))
        if len(Res)==1:
            Res=Res[0]
        return Res
            
    def random(self, obs, samples):
        """
        Sample of parameters from posterior distribution
        """
        res=[]
        yeart=(obs/self.freq).astype("int")
        for year, model  in enumerate(self.modellist):
            temp=model.random(samples)
            if temp.shape[1]<self.maxparam:
                resn=np.array([[np.nan]*self.maxparam]*samples)
                resn[:,:temp.shape[1]]=temp
                temp=resn
            res.append(np.resize(temp,(len(obs[yeart==year]),samples,self.maxparam)))
        #return res
        return np.concatenate(res, axis=0)
    def _BayesianLinearModel__basis(self,obs):
        """
        Expand predictor 
        """
        res=[]
        yeart=(obs/self.freq).astype("int")
        for year,model in enumerate(self.modellist):
            temp=model._BayesianLinearModel__basis(obs[yeart==year][:,np.newaxis])
            if temp.shape[1]<self.maxparam:
                resn=np.zeros((temp.shape[0],self.maxparam))
                resn.fill(np.nan)
                resn[:temp.shape[0],:temp.shape[1]]=temp
                temp=resn
            res.append(temp)
        return np.concatenate(res, axis=0)

def Randomizer(model, nobs, nsamples=1):
    """
    dispatch at the correct class's method
    """
    if model.__class__==ModelList().__class__:
        return model.random(nobs, samples=nsamples)
    else:
        temp=model.random(samples=nsamples)[np.newaxis,:,:]
        return temp
