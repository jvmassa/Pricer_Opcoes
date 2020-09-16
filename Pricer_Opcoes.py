# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:08:15 2020

@author: massa
"""
import numpy as np
import I_Database as idb
import Stats_pack as sp
import datetime as dt
import Aux_functions as af
from scipy.stats import norm 

#constantes empregadas

mu = 0 #média 
sd = 1 #desvio padrao


#Funções de suporte para as equações de precificação, gerenciamento de risco e estimação de volatilidade

#Funções de precificação de opções 
def ds(spot, strike, sigma, TTM, rf, dy = None):
    """
    Inputs:
    spot: valor à vista do ativo-objeto da opção. (float)
    strike: valor de exercício do ativo-objeto para a opção em questão. (float)
    sigma: volatilidade anualizada do ativo-objeto. (float)
    TTM: período até a data de exercício da opção. (float)
    rf: taxa livre de risco contínua. (float)
    dy: dividend yield contínuo do ativo-objeto. (float)
    ---------------------------------------------------------------------------
    Recebe os parâmetros acima e retorna d1 e d2, os termos normalizados empre
    gados nas distribuições de probabilidades acumuladas das equações do modelo
    BSM
    """
    if dy is not None:
        d1 = (np.log(spot/strike) + TTM*((sigma**2)/2 + rf - dy))/(sigma*np.sqrt(TTM))
        d2 = d1 - sigma*np.sqrt(TTM)    
        return (d1,d2)
    else:
        d1 = (np.log(spot/strike) + TTM*((sigma**2)/2 + rf))/(sigma*np.sqrt(TTM))
        d2 = d1 - sigma*np.sqrt(TTM)    
        return d1,d2

def BS_pricer(flag, asset, strike, vcto, daycount ,rf, dy = None):
    """
    Inputs:
    flag: tipo de opção a ser precificada, i.e, call ou put. (string)
    asset: código do ativo-objeto. (string)
    strike: valor de exercício do ativo-objeto para a opção em questão. (float)
    vcto: data do exercício da opção. (string)
    daycount: formato de anualização do TTM. (string)
    rf: taxa livre de risco contínua. (float)
    dy: dividend yield contínuo do ativo-objeto. (float)
    ---------------------------------------------------------------------------
    Recebe os parâmetros acima e retorna o valor, em reais, da opção de um de
    terminado ativo-objeto.
    """
    td = dt.date.today()
    di = af.janela_previa(td,vcto)
    TTM = af.Delta_T(td, vcto, daycount)
    UA = idb.importa_ticker(asset, di, td)['Close']
    sigma = sp.Ann_vol(sp.Log_Ret(UA))
    rf = af.Yd_conv(rf, 'dtoc')
    if flag.lower() == 'call':
        if dy is not None:
            dy = af.Yd_conv(dy,'dtoc')
            d1,d2 = ds(UA[-1],strike, sigma, TTM,rf, dy = dy)
            exd = np.exp(-TTM*dy)
            exr = np.exp(-TTM*rf)
            nd1 = norm.cdf(d1, mu, sd)
            nd2 = norm.cdf(d2, mu, sd)
            BS = UA[-1]*exd*nd1 - strike*exr*nd2
            return round(BS, 6)
        else:
            (d1,d2) = ds(UA[-1],strike, sigma, TTM,rf, dy = dy)            
            exr = np.exp(-TTM*rf)
            nd1 = norm.cdf(d1, mu, sd)
            nd2 = norm.cdf(d2, mu, sd)
            BS = UA[-1]*nd1 - strike*exr*nd2
            return round(BS, 6)              
    elif flag.lower() == 'put':
        if dy is not None:
            dy = af.Yd_conv(dy,'dtoc')
            (d1,d2) = ds(UA[-1],strike, sigma, TTM,rf, dy = dy)
            exd = np.exp(-TTM*dy)
            exr = np.exp(-TTM*rf)
            nd1 = norm.cdf(d1, mu, sd)
            nd2 = norm.cdf(d2, mu, sd)          
            BS = strike*exr*nd2 - UA[-1]*exd*nd1
            return round(BS,6)
        else:
            (d1,d2) = ds(UA[-1],strike, sigma, TTM,rf, dy = dy)            
            exr = np.exp(-TTM*rf)
            nd1 = norm.cdf(-d1, mu, sd)
            nd2 = norm.cdf(-d2, mu, sd)
            BS = strike*exr*nd2 - UA[-1]*nd1
            return round(BS,6)
        
def Delta_BS(flag, asset, strike, vcto, daycount ,rf, dy = None):
    """
    Inputs:
    flag: tipo de opção precificada, i.e, call ou put. (string)
    asset: código do ativo-objeto. (string)
    strike: valor de exercício do ativo-objeto para a opção em questão. (float)
    vcto: data do exercício da opção. (string)
    daycount: formato de anualização do TTM. (string)
    rf: taxa livre de risco contínua. (float)
    dy: dividend yield contínuo do ativo-objeto. (float)
    ---------------------------------------------------------------------------
    Recebe os parâmetros acima e retorna o valor, em reais, do delta de uma de
    terminada opção.
    """
    td = dt.date.today()
    di = af.janela_previa(td,vcto)
    TTM = af.Delta_T(td, vcto, daycount)
    UA = idb.importa_ticker(asset, di, td)['Close']
    sigma = sp.Ann_vol(sp.Log_Ret(UA))
    rf = af.Yd_conv(rf, 'dtoc')
    if flag.lower() =='call':
        if dy is not None:
            dy = af.Yd_conv(dy,'dtoc')
            (d1,d2) = ds(UA[-1],strike, sigma, TTM,rf, dy = dy)
            exd = np.exp(-TTM*dy)
            nd1 = norm.cdf(d1, mu, sd)
            Delta = exd*nd1
            return round(Delta,6)
        else:
            (d1,d2) = ds(UA[-1],strike, sigma, TTM,rf, dy = dy)
            nd1 = norm.cdf(d1, mu, sd)
            Delta = nd1
            return round(Delta,6)
    elif flag.lower() == 'put':
        if dy is not None:
            dy = af.Yd_conv(dy,'dtoc')
            (d1,d2) = ds(UA[-1],strike, sigma, TTM,rf, dy = dy)
            exd = np.exp(-TTM*dy)
            nd1 = norm.cdf(-d1, mu, sd)
            Delta = -exd*nd1
            return round(Delta,6)
        else:
            (d1,d2) = ds(UA[-1],strike, sigma, TTM,rf, dy = dy)
            nd1 = norm.cdf(-d1, mu, sd)
            Delta = -nd1
            return round(Delta,6)

def Gamma_BS(asset, strike, vcto, daycount ,rf, dy = None):
    """
    Inputs:    
    asset: código do ativo-objeto. (string)
    strike: valor de exercício do ativo-objeto para a opção em questão. (float)
    vcto: data do exercício da opção. (string)
    daycount: formato de anualização do TTM. (string)
    rf: taxa livre de risco contínua. (float)
    dy: dividend yield contínuo do ativo-objeto. (float)
    ---------------------------------------------------------------------------
    Recebe os parâmetros acima e retorna o valor, em reais, do gamma de uma de
    terminada opção.
    """
    td = dt.date.today()
    di = af.janela_previa(td,vcto)
    TTM = af.Delta_T(td, vcto, daycount)
    UA = idb.importa_ticker(asset, di, td)['Close']
    sigma = sp.Ann_vol(sp.Log_Ret(UA))
    rf = af.Yd_conv(rf, 'dtoc')
    if dy is not None:
        dy = af.Yd_conv(dy,'dtoc')
        (d1,d2) = ds(UA[-1],strike, sigma, TTM,rf, dy = dy)
        exd = np.exp(-TTM*dy)
        nd1 = norm.cdf(d1, mu, sd)
        Gamma = (exd)*(nd1/UA[-1]*sigma*np.sqrt(TTM))
        return round(Gamma,6)
    else:
        (d1,d2) = ds(UA[-1],strike, sigma, TTM,rf, dy = dy)
        nd1 = norm.cdf(d1, mu, sd)
        Gamma = nd1/UA[-1]*sigma*np.sqrt(TTM)
        return round(Gamma,6)
    
def Vega_BS(asset, strike, vcto, daycount ,rf, dy = None):
    """
    Inputs:    
    asset: código do ativo-objeto. (string)
    strike: valor de exercício do ativo-objeto para a opção em questão. (float)
    vcto: data do exercício da opção. (string)
    daycount: formato de anualização do TTM. (string)
    rf: taxa livre de risco contínua. (float)
    dy: dividend yield contínuo do ativo-objeto. (float)
    ---------------------------------------------------------------------------
    Recebe os parâmetros acima e retorna o valor, em reais, do vega de 100bps
    de uma determinada opção.
    """
    td = dt.date.today()
    di = af.janela_previa(td,vcto)
    TTM = af.Delta_T(td, vcto, daycount)
    UA = idb.importa_ticker(asset, di, td)['Close']
    sigma = sp.Ann_vol(sp.Log_Ret(UA))
    rf = af.Yd_conv(rf, 'dtoc')
    if dy is not None:
        dy = af.Yd_conv(dy,'dtoc')
        (d1,d2) = ds(UA[-1],strike, sigma, TTM,rf, dy = dy)
        exd = np.exp(-TTM*dy)
        nd1 = norm.cdf(d1, mu, sd)
        Vega = (UA[-1]*np.sqrt(TTM)*nd1)/exd
        return round(Vega/100,6)
    else:
        (d1,d2) = ds(UA[-1],strike, sigma, TTM,rf, dy = dy)
        nd1 = norm.cdf(d1, mu, sd)
        Vega = UA[-1]*np.sqrt(TTM)*nd1
        return round(Vega/100,6)
    
def Theta_BS(flag, asset, strike, vcto, daycount ,rf, dy = None):
    """
    Inputs:
    flag: tipo de opção precificada, i.e, call ou put. (string)
    asset: código do ativo-objeto. (string)
    strike: valor de exercício do ativo-objeto para a opção em questão. (float)
    vcto: data do exercício da opção. (string)
    daycount: formato de anualização do TTM. (string)
    rf: taxa livre de risco contínua. (float)
    dy: dividend yield contínuo do ativo-objeto. (float)
    ---------------------------------------------------------------------------
    Recebe os parâmetros acima e retorna o valor, em reais, do theta de um dia 
    de uma determinada opção.
    """    
    td = dt.date.today()
    di = af.janela_previa(td,vcto)
    TTM = af.Delta_T(td, vcto, daycount)
    UA = idb.importa_ticker(asset, di, td)['Close']
    sigma = sp.Ann_vol(sp.Log_Ret(UA))
    rf = af.Yd_conv(rf, 'dtoc')
    if flag.lower() == 'call':
        if dy is not None:
            dy = af.Yd_conv(dy,'dtoc')
            (d1,d2) = ds(UA[-1],strike, sigma, TTM,rf, dy = dy)
            exd = np.exp(-TTM*dy)
            exr = np.exp(-TTM*rf)
            nd1 = norm.cdf(d1, mu, sd)
            nd2 = norm.cdf(d2, mu, sd)
            theta_vol = -(UA[-1]*nd1*sigma*exr)/2*np.sqrt(TTM) + (dy*UA[-1]*nd1*exd)
            theta_juro = -rf*strike*exr*nd2
            Theta = theta_vol + theta_juro
            return round(Theta/252,6)
        else:
            (d1,d2) = ds(UA[-1],strike, sigma, TTM,rf, dy = dy)
            exr = np.exp(-TTM*rf)
            nd1 = norm.cdf(d1, mu, sd)
            nd2 = norm.cdf(d2, mu, sd)
            theta_vol = -(UA[-1]*nd1*sigma*exr)/2*np.sqrt(TTM)
            theta_juro = -rf*strike*exr*nd2
            Theta = theta_vol + theta_juro
            return round(Theta/252,6)
    if flag.lower() == 'put':
        if dy is not None:
            dy = af.Yd_conv(dy,'dtoc')
            (d1,d2) = ds(UA[-1],strike, sigma, TTM,rf, dy = dy)
            exd = np.exp(-TTM*dy)
            exr = np.exp(-TTM*rf)
            nd1 = norm.cdf(d1, mu, sd)
            nd1_1 = norm.cdf(-d1, mu, sd)
            nd2 = norm.cdf(-d2, mu, sd)
            theta_vol = -(UA[-1]*nd1*sigma*exr)/2*np.sqrt(TTM) + (dy*UA[-1]*nd1_1*exd)
            theta_juro = -rf*strike*exr*nd2
            Theta = theta_vol + theta_juro
            return round(Theta/252,6)
        else:
            (d1,d2) = ds(UA[-1],strike, sigma, TTM,rf, dy = dy)
            exr = np.exp(-TTM*rf)
            nd1 = norm.cdf(d1, mu, sd)
            nd2 = norm.cdf(-d2, mu, sd)
            theta_vol = -(UA[-1]*nd1*sigma*exr)/2*np.sqrt(TTM)
            theta_juro = -rf*strike*exr*nd2
            Theta = theta_vol + theta_juro
            return round(Theta/252,6)
        
def Rho_BS(flag, asset, strike, vcto, daycount ,rf, dy = None):
    """
    Inputs:
    flag: tipo de opção precificada, i.e, call ou put. (string)
    asset: código do ativo-objeto. (string)
    strike: valor de exercício do ativo-objeto para a opção em questão. (float)
    vcto: data do exercício da opção. (string)
    daycount: formato de anualização do TTM. (string)
    rf: taxa livre de risco contínua. (float)
    dy: dividend yield contínuo do ativo-objeto. (float)
    ---------------------------------------------------------------------------
    Recebe os parâmetros acima e retorna o valor, em reais, do rho de 100bps
    de uma determinada opção.
    """
    td = dt.date.today()
    di = af.janela_previa(td,vcto)
    TTM = af.Delta_T(td, vcto, daycount)
    UA = idb.importa_ticker(asset, di, td)['Close']
    sigma = sp.Ann_vol(sp.Log_Ret(UA))
    rf = af.Yd_conv(rf, 'dtoc')
    if flag.lower() == 'call':
        if dy is not None:
            (d1,d2) = ds(UA[-1],strike, sigma, TTM,rf, dy = dy)
            exr = np.exp(-TTM*rf)            
            nd2 = norm.cdf(d2, mu, sd)
            Rho = (strike*TTM*exr*nd2)/100
            return round(Rho,6)
        else:
            (d1,d2) = ds(UA[-1],strike, sigma, TTM,rf, dy = dy)
            exr = np.exp(-TTM*rf)            
            nd2 = norm.cdf(d2, mu, sd)
            Rho = (strike*TTM*exr*nd2)/100
            return round(Rho,6)
    elif flag.lower() == 'put':
        if dy is not None:
            (d1,d2) = ds(UA[-1],strike, sigma, TTM,rf, dy = dy)
            exr = np.exp(-TTM*rf)            
            nd2 = norm.cdf(-d2, mu, sd)
            Rho = -(strike*TTM*exr*nd2)/100
            return round(Rho,6)
        else:
            (d1,d2) = ds(UA[-1],strike, sigma, TTM,rf, dy = dy)
            exr = np.exp(-TTM*rf)            
            nd2 = norm.cdf(-d2, mu, sd)
            Rho = -(strike*TTM*exr*nd2)/100
            return round(Rho,6)
        
    
        
    
        
            
    

    
    
        
        
    
    

    
        
        
        
        
            
    
    
        
    