"""Prepare o recorte real do SP156. Nao altera nem inventa resultados.

Uso: python preparar_dados_sp156.py --fonte caminho/arquivo-original.csv
Sem --fonte, baixa o CSV oficial (aproximadamente 155 MB) para .cache_sp156/.
O CSV pequeno resultante e a base efetivamente usada no notebook.
"""
from pathlib import Path
import argparse,hashlib,json,urllib.request
import pandas as pd

BASE=Path(__file__).resolve().parent
URL='https://dados.prefeitura.sp.gov.br/dataset/0aecfa2b-aa3a-40d4-8183-0d4351b7fd0a/resource/7c5aeb58-2cad-40ba-9e9e-af1c17cc5070/download/arquivo-final-1-trim-25.csv'
SHA256_ORIGINAL='f9d02a8bf358f019a5851e0d3d05daa236f803f5b22f254b66fd7b69f63e5170'
PORTAL='https://dados.prefeitura.sp.gov.br/dataset/dados-do-sp156'
DICIONARIO='https://dados.prefeitura.sp.gov.br/dataset/0aecfa2b-aa3a-40d4-8183-0d4351b7fd0a/resource/e5c102e2-f27a-4e0a-ae99-5948abecff0e/download/dicionario-dados-abertos-sp156-2020-em-diante.txt'
SEMENTE=20261117
TEMAS=['Rua e bairro','Lixo e limpeza']
N_AMOSTRA=60000
PRAZO=7
# Ultima modificacao HTTP do recurso, nao data exata de extracao/publicacao.
# A fonte registra fechamentos ate agosto/2026, muito apos o trimestre.
REFERENCIA_ARQUIVO=pd.Timestamp('2026-08-10')

def preparar(source):
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!=SHA256_ORIGINAL:
        raise ValueError('A fonte foi republicada ou mudou. Confira o recurso e atualize a auditoria antes de continuar.')
    raw=pd.read_csv(source,sep=';',encoding='cp1252',dtype=str,keep_default_na=False)
    raw_n=len(raw);empty=int(raw.apply(lambda col:col.str.strip().eq('')).all(axis=1).sum())
    selected=raw.loc[raw.Tema.isin(TEMAS)].copy()
    selected['id_registro']=['SP156-2025T1-'+str(i+2) for i in selected.index]
    count_before=len(selected)
    selected=selected.rename(columns={'Data de Abertura':'data_abertura','Data de Finalização':'data_finalizacao','Meio de Comunicação':'canal','Serviço':'servico','Distrito':'distrito','Status':'status','Tema':'tema'})
    for col in ['canal','servico','distrito','status','tema']:
        selected[col]=selected[col].str.strip()
    a=pd.to_datetime(selected.data_abertura,format='%d/%m/%Y',errors='coerce')
    f=pd.to_datetime(selected.data_finalizacao,format='%d/%m/%Y',errors='coerce')
    delta=(f-a).dt.days
    bad_open=a.isna()|~a.between('2025-01-01','2025-03-31')
    bad_close=(selected.data_finalizacao.ne('')&f.isna())|(delta<0)|(f>REFERENCIA_ARQUIVO)
    unknown=selected.status.isin(['Finalizada','Cancelada','Indeferido','Indeferida'])&f.isna()
    immature=a+pd.Timedelta(days=PRAZO)>REFERENCIA_ARQUIVO
    bad=bad_open|bad_close|unknown|immature
    audit={'linhas_csv_original':raw_n,'linhas_totalmente_vazias':empty,'recorte_por_tema_antes_limpeza':count_before,
           'exclusoes_data_abertura_invalida':int(bad_open.sum()),'exclusoes_data_finalizacao_invalida':int(bad_close.sum()),
           'exclusoes_terminal_sem_data':int(unknown.sum()),'exclusoes_janela_imatura':int(immature.sum()),
           'exclusoes_unicas':int(bad.sum())}
    selected=selected.loc[~bad].copy();a=a.loc[~bad];f=f.loc[~bad]
    selected['data_abertura']=a
    selected['data_finalizacao']=f
    selected['dias_ate_encerramento']=(f-a).dt.days.astype('Int64')
    selected['encerramento_7d']=selected.dias_ate_encerramento.between(0,PRAZO).fillna(False).astype('int64')
    selected['dia_semana']=a.dt.dayofweek.astype('int64')
    selected['distrito']=selected.distrito.mask(selected.distrito.eq(''),'NAO INFORMADO')
    selected['canal']=selected.canal.mask(selected.canal.eq(''),'NAO INFORMADO')
    if selected.servico.eq('').any():
        raise ValueError('Servico em branco no recorte; revisar em vez de excluir silenciosamente.')
    audit['recorte_valido']=len(selected)
    audit['classes_recorte_completo']={str(k):int(v) for k,v in selected.encerramento_7d.value_counts().sort_index().items()}
    audit['status_recorte_completo']=selected.status.value_counts().to_dict()
    sample=selected.sample(n=min(N_AMOSTRA,len(selected)),random_state=SEMENTE).sort_values(['data_abertura','id_registro']).reset_index(drop=True)
    columns=['id_registro','data_abertura','data_finalizacao','tema','servico','distrito','canal','dia_semana','status','dias_ate_encerramento','encerramento_7d']
    sample=sample[columns]
    sample.to_csv(BASE/'dados_resolveai.csv',index=False,date_format='%Y-%m-%d',encoding='utf-8')
    audit['amostra_n']=len(sample)
    audit['classes_amostra']={str(k):int(v) for k,v in sample.encerramento_7d.value_counts().sort_index().items()}
    audit['datas_amostra']={'inicio':str(sample.data_abertura.min().date()),'fim':str(sample.data_abertura.max().date())}
    audit['amostragem']='Amostra uniforme sem reposicao de 60000 registros validos, random_state=20261117, sem estratificacao por alvo. Contagens repetidas de pedidos nao sao removidas por igualdade de entradas.'
    audit['sem_enderecos_pessoais']='Nao exportamos logradouro, numero, CEP, coordenadas ou campos pessoais. id_registro e uma referencia a linha no CSV publico, nao numero de protocolo.'
    source_meta={'nome':'Prefeitura de Sao Paulo - SP156, solicitacoes 1o trimestre de 2025','url':PORTAL,'csv_url':URL,'dicionario_url':DICIONARIO,'sha256_original':digest,'sha256_csv_usado':hashlib.sha256((BASE/'dados_resolveai.csv').read_bytes()).hexdigest(),
                 'download_em':'2026-11-17','ultima_modificacao_http_em':'2026-08-10','extracao_exata':'Nao informada no recurso. Last-Modified indica modificacao HTTP, nao necessariamente extracao ou publicacao. O arquivo registra fechamentos ate agosto/2026; as aberturas do trimestre sao de janeiro a marco/2025.',
                 'periodo':'Aberturas de janeiro a marco de 2025; amostra de 60.000 solicitacoes.',
                 'temas':TEMAS,'prazo_dias':PRAZO,'dados_sinteticos':False,'auditoria':audit,
                 'limitacoes':['As datas da fonte nao incluem horarios; o alvo usa diferenca entre datas em dias corridos.',
                               'Encerramento administrativo inclui cancelamentos com data de fechamento; nao comprova resolucao fisica.',
                               'O servico publicado pode ter sido reclassificado depois da abertura. A avaliacao e retrospectiva, sem validacao prospectiva da classificacao original.',
                               'O CSV publico nao fornece identificador de protocolo; pedidos com os mesmos atributos podem ser solicitacoes distintas.',
                               'A fonte pode ser republicada; o hash fixa a versao utilizada. Resultados sao do recorte Rua e bairro/Lixo e limpeza em Sao Paulo, nao de outras cidades.']}
    (BASE/'fonte_dados.json').write_text(json.dumps(source_meta,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(source_meta,ensure_ascii=True),flush=True)
    return source_meta

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--fonte',type=Path)
    args=parser.parse_args();path=args.fonte
    if path is None:
        cache=BASE/'.cache_sp156';cache.mkdir(exist_ok=True);path=cache/'sp156_2025_t1_original.csv'
        if not path.exists():urllib.request.urlretrieve(URL,path)
    preparar(path)
