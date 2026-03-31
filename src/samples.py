import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax 
import gc
from model_jax import Params, utility
from mh_jax import mh_sample
from tqdm import tqdm

q=pd.read_parquet('C:\BASIC_LEARNING\mls\code\data\q_d\q_d_d_f.parquet')
p=pd.read_parquet("C:\BASIC_LEARNING\mls\code\data\p_d\p_d_d_f.parquet")
L=q.shape[1]-2
p_l=p.iloc[-1,-L:]
week=p_l.name
q_init=np.zeros(L)
p_np=p_l.values
del p,q 
gc.collect()
seed=42
key=jax.random.PRNGKey(seed)
n_samples=150_000
ws=[0,0.25,0.5,0.75,1]
ks=[2,5,10]
for k in tqdm(ks):
    for weight in tqdm(ws):
        data=np.load(f'C:\BASIC_LEARNING\mls\code\JAX\Parameters\model_K{k}_w{weight}.npz')
        params=Params(
        A=jnp.array(data['A'],dtype=jnp.float32),
        b=jnp.array(data['b'],dtype=jnp.float32),
        delta=jnp.array(data['delta'],dtype=jnp.float32)
        )
        samples=mh_sample(key,params,q_init,p_np,n_samples)
        samples=np.array(samples)
        np.save(f'C:\BASIC_LEARNING\mls\code\JAX\samples\samples_K{k}_w{weight}.npy',samples)