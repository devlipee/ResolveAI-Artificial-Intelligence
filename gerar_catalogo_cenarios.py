"""Gere cinco exemplos de chance baixa e cinco de chance alta para demonstracao.

A cobertura e conferida no treino e na validacao antes de formar as faixas.
Dentro de cada faixa, o ranking usa somente contagens, nunca acertos no teste
nem o valor mais extremo de probabilidade. Os pisos sao regras operacionais,
nao garantia estatistica. Este catalogo e demonstrativo, nao um novo benchmark.
Executar depois de exportar os modelos: python gerar_catalogo_cenarios.py
"""
from pathlib import Path
import hashlib,json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,brier_score_loss,confusion_matrix

BASE=Path(__file__).resolve().parent
POLITICA={
    'min_treino_servico':200,
    'min_treino_combinacao':8,
    'min_validacao_combinacao':2,
    'min_treino_contexto':20,
    'min_validacao_servico':50,
    'min_acuracia_validacao_ambos':.75,
    'exemplos_por_faixa':5,
}
DIAS=['segunda-feira','terca-feira','quarta-feira','quinta-feira','sexta-feira','sabado','domingo']
MODELOS=['regressao_logistica','random_forest']
FAIXAS_CHANCE={'baixa_max':.30,'alta_min':.70}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def metricas(y,p,prob):
    return dict(acuracia=float(accuracy_score(y,p)),
                f1_macro=float(f1_score(y,p,labels=[0,1],average='macro',zero_division=0)),
                brier=float(brier_score_loss(y,prob)),
                matriz=confusion_matrix(y,p,labels=[0,1]).tolist())

def gerar():
    modelpath=BASE/'modelos_resolveai.joblib';csvpath=BASE/'dados_resolveai.csv'
    b=joblib.load(modelpath);features=b['entradas'];csv_sha=sha(csvpath)
    assert csv_sha==b['metadados']['csv_sha256']
    d=pd.read_csv(csvpath,parse_dates=['data_abertura'])
    treino=d.loc[d.data_abertura<'2025-02-15'].copy()
    val=d.loc[(d.data_abertura>='2025-02-22')&(d.data_abertura<'2025-03-09')].copy()
    for name in MODELOS:
        val[name+'_pred']=b[name].predict(val[features])
        val[name+'_prob']=b[name].predict_proba(val[features])[:,1]
    por_servico=treino.groupby('servico').encerramento_7d.agg(['size','mean'])
    validacao_servico={}
    for servico,g in val.groupby('servico'):
        ntreino=int(por_servico.loc[servico,'size']) if servico in por_servico.index else 0
        if ntreino<POLITICA['min_treino_servico'] or len(g)<POLITICA['min_validacao_servico']:continue
        scores={name:metricas(g.encerramento_7d,g[name+'_pred'],g[name+'_prob']) for name in MODELOS}
        if any(m['acuracia']<POLITICA['min_acuracia_validacao_ambos'] for m in scores.values()):continue
        validacao_servico[servico]={'treino_n':ntreino,'validacao_n':len(g),'metricas':scores,
            'classes':{str(k):int(v) for k,v in g.encerramento_7d.value_counts().reindex([0,1],fill_value=0).items()}}
    val_counts=val.groupby(features).size()
    campos_contexto=features[:-1]  # servico, distrito e canal; sem o dia da semana
    treino_contextos=treino.groupby(campos_contexto).size()
    validacao_contextos=val.groupby(campos_contexto).size()
    cenarios=[]
    for key,n in treino.groupby(features).size().items():
        servico=key[0]
        nval=int(val_counts.get(key,0))
        chave_contexto=key[:-1]
        ncontexto=int(treino_contextos.get(chave_contexto,0))
        if (n<POLITICA['min_treino_combinacao']
                or nval<POLITICA['min_validacao_combinacao']
                or ncontexto<POLITICA['min_treino_contexto']
                or servico not in validacao_servico):continue
        c=dict(zip(features,key));c['dia_semana']=int(c['dia_semana'])
        canonical=json.dumps(c,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        sid='sp156-'+hashlib.sha256(canonical.encode('utf8')).hexdigest()[:20]
        short='Remocao de entulho' if servico.startswith('Entulho e grandes objetos') else 'Poluicao sonora (PSIU)' if servico.endswith('PSIU') else servico
        name=f"{short} - {c['distrito']} - {c['canal']} - {DIAS[c['dia_semana']]}"
        s=validacao_servico[servico]
        cenarios.append(dict(id=sid,nome=name,caracteristicas=c,
            cobertura={'treino_combinacao':int(n),'validacao_combinacao':nval,
                       'treino_contexto':ncontexto,
                       'validacao_contexto':int(validacao_contextos.get(chave_contexto,0)),
                       'treino_servico':s['treino_n'],'validacao_servico':s['validacao_n']},
            metricas_validacao_servico=s['metricas'],classes_validacao_servico=s['classes']))
    def ranking(cenario):
        return (-cenario['cobertura']['treino_combinacao'],
                -cenario['cobertura']['validacao_combinacao'],cenario['id'])
    cenarios.sort(key=ranking)
    if not cenarios:raise ValueError('Nenhum cenario atende a politica. Revisar o produto sem recorrer ao teste para escolher casos.')
    # Faixas respondem ao pedido de cinco exemplos baixos e cinco altos.
    # O valor da probabilidade nao entra no ranking dentro de cada faixa.
    contexto=pd.DataFrame([s['caracteristicas'] for s in cenarios],columns=features)
    probabilidades={name:b[name].predict_proba(contexto)[:,1] for name in MODELOS}
    for i,s in enumerate(cenarios):
        probs={name:float(probabilidades[name][i]) for name in MODELOS}
        s['probabilidades_estimadas']=probs
        s['faixa_chance']=('baixa' if max(probs.values())<=FAIXAS_CHANCE['baixa_max'] else
                           'alta' if min(probs.values())>=FAIXAS_CHANCE['alta_min'] else 'intermediaria')
    candidatos_por_faixa={faixa:sum(s['faixa_chance']==faixa for s in cenarios)
                          for faixa in ('baixa','alta','intermediaria')}
    selecionados=[]
    for faixa in ('baixa','alta'):
        candidatos=sorted((s for s in cenarios if s['faixa_chance']==faixa),key=ranking)
        if len(candidatos)<POLITICA['exemplos_por_faixa']:
            raise ValueError(f"Ha somente {len(candidatos)} exemplos da faixa {faixa}; "
                             "nao reduza os criterios automaticamente para preencher a demonstracao.")
        selecionados.extend(candidatos[:POLITICA['exemplos_por_faixa']])
    cenarios=selecionados
    selecao_demonstracao={
        'modo':'cinco_por_faixa',
        'candidatos_por_faixa':candidatos_por_faixa,
        'quantidade_por_faixa':POLITICA['exemplos_por_faixa'],
        'ranking':['treino_combinacao decrescente','validacao_combinacao decrescente','id crescente'],
        'nota':'Exemplos selecionados para demonstracao por faixa e cobertura. '
               'A probabilidade define a faixa; seu valor nao desempata candidatos. '
               'Nao usamos acertos no teste nem desempenho da combinacao para rankear.',
    }
    # The complete list is frozen here, before any evaluation of the final test.
    tuples={tuple(c['caracteristicas'][f] for f in features) for c in cenarios}
    assinatura=hashlib.sha256(json.dumps({'politica':POLITICA,'ids':sorted(c['id'] for c in cenarios)},sort_keys=True).encode()).hexdigest()
    def avaliar(frame):
        mask=pd.Series([tuple(row) in tuples for row in frame[features].itertuples(index=False,name=None)],index=frame.index)
        selected=frame.loc[mask].copy();n=len(selected)
        result={'n':n,'total_conjunto':len(frame),'cobertura':n/len(frame),
            'classes':{str(k):int(v) for k,v in selected.encerramento_7d.value_counts().reindex([0,1],fill_value=0).items()},'metricas':{}}
        if not n:return result
        for name in MODELOS:
            result['metricas'][name]=metricas(selected.encerramento_7d,b[name].predict(selected[features]),b[name].predict_proba(selected[features])[:,1])
        # Same global majority reference as the original model evaluation.
        global_prob=float(treino.encerramento_7d.mean())
        result['metricas']['referencia_dummy']=metricas(selected.encerramento_7d,[int(global_prob>=.5)]*n,[global_prob]*n)
        # Service-level majority reveals when concentrated outcomes explain accuracy.
        service_probs=selected.servico.map(por_servico['mean']).to_numpy(dtype=float)
        result['metricas']['referencia_por_servico']=metricas(selected.encerramento_7d,(service_probs>=.5).astype(int),service_probs)
        return result
    teste=d.loc[d.data_abertura>='2025-03-16'].copy()
    avaliacao={'validacao':avaliar(val),'teste':avaliar(teste)}
    catalogo={'schema':1,'entradas':features,'modelo_sha256':sha(modelpath),'csv_sha256':csv_sha,
        'politica':POLITICA,'assinatura_selecao':assinatura,'cenarios':cenarios,'avaliacao':avaliacao,
        'faixas_chance':FAIXAS_CHANCE,
        'selecao_demonstracao':selecao_demonstracao,
        'nota_selecao':'Catalogo demonstrativo com cinco exemplos baixos e cinco altos, '
                       'filtrados por cobertura e avaliacao do servico na validacao. '
                       'Ranking por contagens de treino e validacao, com IDs congelados antes '
                       'de avaliar o recorte do teste. A avaliacao do servico nao equivale '
                       'a avaliacao da combinacao exata.',
        'nota_interpretacao':'Os indicadores deste recorte selecionado para demonstracao sao '
                             'descritivos, nao uma nova validacao independente nem substitutos '
                             'das metricas globais. Servicos possuem classes concentradas e a '
                             'referencia por servico pode empatar com os modelos. Cobertura '
                             'e desempenho historico nao garantem acerto, calibracao nem '
                             'concordancia em uma nova solicitacao.'}
    (BASE/'catalogo_cenarios.json').write_text(json.dumps(catalogo,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'cenarios':len(cenarios),'servicos':sorted(validacao_servico.keys() & {c['caracteristicas']['servico'] for c in cenarios}),'politica':POLITICA,'avaliacao':avaliacao},ensure_ascii=True),flush=True)
    return catalogo

if __name__=='__main__':gerar()
