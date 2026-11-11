from copy import deepcopy

import fitz
import pytest

from app.services.table_native_financial import refine_financial_boundaries


def add(page, cells, row, col, text, x, y, header=False, span=1):
    before=len(page.get_text('words'))
    page.insert_text((x,y),text,fontsize=8)
    words=page.get_text('words')[before:]
    b=fitz.Rect(min(w[0] for w in words),min(w[1] for w in words),max(w[2] for w in words),max(w[3] for w in words))
    cell=dict(page_index=7,row_index=row,column_index=col,row_span=1,column_span=span,
              text=text,raw_text=text,is_header=header,source='docling_tableformer',
              bbox=dict(x0=b.x0,y0=b.y0,x1=b.x1,y1=b.y1),geometry_status='available')
    cells.append(cell)
    return cell


def table(cells, rows=3, cols=3, bottom=160):
    grid=[['']*cols for _ in range(rows)]
    for c in cells:grid[c['row_index']][c['column_index']]=c['text']
    return dict(page_index=7,page_width=700,page_height=400,
                bbox=dict(x0=10,y0=20,x1=680,y1=bottom),cells=cells,rows=grid,
                row_count=rows,col_count=cols,source='docling_tableformer',warnings=[])


def header_fixture(extra=False):
    doc=fitz.open();p=doc.new_page(width=700,height=400);cells=[]
    add(p,cells,0,0,'Proceeds (Box 1d)',20,40,True)
    add(p,cells,0,1,'Quantity',240,40,True)
    a=add(p,cells,0,2,'Wash Sale Loss Disallowed (Box 1g)',360,40,True)
    b=add(p,cells,0,2,'Proceeds from C/Q',550,40,True)
    a['text']=a['raw_text']=a['text']+' '+b['text'];a['bbox']['x1']=b['bbox']['x1'];cells.pop()
    for row in [1,2]:
        add(p,cells,row,0,'100.00',20,40+row*30)
        add(p,cells,row,1,'2',240,40+row*30)
        add(p,cells,row,2,'0.00',430,40+row*30)
    if extra:p.insert_text((550,70),'999.00',fontsize=8)
    return doc,table(cells)


def test_split_only_proven_empty_terminal_column():
    doc,t=header_fixture();before=deepcopy(t)
    result,u=refine_financial_boundaries(doc[0],[t]);r=result[0]
    assert u==0 and r['col_count']==4
    assert r['rows'][0][-2:]==['Wash Sale Loss Disallowed (Box 1g)','Proceeds from C/Q']
    assert all(row[-1]=='' for row in r['rows'][1:])
    assert t==before and all(c['page_index']==7 for c in r['cells'])


@pytest.mark.parametrize('change',['populated','fabricated','hidden','body_span'])
def test_uncertain_terminal_column_is_not_split(change):
    doc,t=header_fixture(extra=change=='populated')
    if change=='fabricated':t['cells'][-1]['raw_text']='999.00'
    if change=='hidden':doc[0].insert_text((20,200),'HIDDEN',render_mode=3)
    if change=='body_span':t['cells'][-1]['column_span']=2
    result,u=refine_financial_boundaries(doc[0],[t])
    assert result[0]['col_count']==3
    assert u==1


def rule_fixture(value='123.45',dashes=100):
    doc=fitz.open();p=doc.new_page(width=700,height=400);cells=[]
    add(p,cells,0,0,'Proceeds (Box 1d)',20,40,True)
    c=add(p,cells,1,0,value,250,90)
    for x in range(dashes):p.insert_text((20+x*6,100),'-',fontsize=5)
    # The model wrongly includes a portion of the separator in the amount.
    selected=[w for w in p.get_text('words') if 245<(w[0]+w[2])/2<300 and w[4]=='-']
    c['text']=c['raw_text']=value+' '+' '.join(w[4] for w in selected)
    c['bbox'].update(x0=245,x1=300,y1=102)
    return doc,table(cells,2,1),c


def test_separator_repair_preserves_negative_sign_and_original():
    doc,t,c=rule_fixture('-123.45');before=deepcopy(t)
    result,u=refine_financial_boundaries(doc[0],[t])
    assert result[0]['rows'][1][0]=='-123.45'
    assert result[0]['cells'][1]['bbox']['y1']<100
    assert t==before and u==0


@pytest.mark.parametrize('change',['short_rule','missing_sign','same_line_number'])
def test_unproved_minus_is_never_removed(change):
    doc,t,c=rule_fixture('-123.45',10 if change=='short_rule' else 100)
    if change=='missing_sign':c['raw_text']=c['raw_text'].replace('-123.45','123.45')
    if change=='same_line_number':doc[0].insert_text((630,100),'888',fontsize=5)
    before=deepcopy(c)
    result,_=refine_financial_boundaries(doc[0],[t])
    assert result[0]['cells'][1]==before


def summary_fixture(target='Box B Short-Term Realized Loss',extra=False,loss_value='0.00'):
    doc=fitz.open();p=doc.new_page(width=700,height=400);cells=[]
    add(p,cells,0,0,'Proceeds (Box 1d)',20,40,True)
    add(p,cells,0,1,'Gain/Loss (-)',540,40,True)
    add(p,cells,1,0,'Box B Short-Term Realized Gain',20,80)
    add(p,cells,1,1,'100.00',575,80)
    p.insert_text((20,92),target,fontsize=8)
    # Align right edges, not left edges, of the differently sized amounts.
    right=cells[-1]['bbox']['x1']
    p.insert_text((right-fitz.get_text_length(loss_value,fontsize=8),92),loss_value,fontsize=8)
    if extra:p.insert_text((350,92),'OTHER',fontsize=8)
    return doc,table(cells,2,2,110)


def test_source_loss_summary_recovers_once_and_keeps_page():
    doc,t=summary_fixture();before=deepcopy(t)
    result,u=refine_financial_boundaries(doc[0],[t]);r=result[0]
    assert r['row_count']==3 and r['rows'][-1]==['Box B Short-Term Realized Loss','0.00']
    assert all(c['page_index']==7 for c in r['cells']) and u==0
    again,u=refine_financial_boundaries(doc[0],result)
    assert again==result and t==before


@pytest.mark.parametrize('change',['other_box','extra_word','wrong_header','far_away','already_assigned'])
def test_summary_requires_same_box_column_and_unique_source(change):
    doc,t=summary_fixture('Box C Short-Term Realized Loss' if change=='other_box' else 'Box B Short-Term Realized Loss',change=='extra_word')
    if change=='wrong_header':t['cells'][1]['text']='Quantity'
    if change=='far_away':t['bbox']['y1']=85
    if change=='already_assigned':t['cells'][1]['bbox']=dict(x0=10,y0=85,x1=650,y1=100)
    result,_=refine_financial_boundaries(doc[0],[t])
    assert result[0]['row_count']==2


def test_header_prose_in_transaction_quantity_requires_review():
    doc=fitz.open();p=doc.new_page(width=700,height=400);cells=[]
    add(p,cells,0,0,'Proceeds (Box 1d)',20,40,True)
    add(p,cells,0,1,'Quantity',240,40,True)
    add(p,cells,1,0,'100.00',20,100)
    c=add(p,cells,1,1,'HEADER',240,70)
    b=add(p,cells,1,1,'2',240,100);cells.pop()
    c['text']=c['raw_text']='HEADER 2';c['bbox']['y1']=b['bbox']['y1']
    result,u=refine_financial_boundaries(p,[table(cells,2,2)])
    assert u==1 and 'native_financial_row_contaminated' in result[0]['warnings']


def test_adapter_keeps_recovered_summary_on_its_original_document_page(monkeypatch):
    from types import SimpleNamespace as NS
    from app.services import table_docling
    one,t=summary_fixture(loss_value='0.00')
    two,_=summary_fixture(loss_value='55.00')
    combined=fitz.open();combined.new_page();combined.insert_pdf(one);combined.insert_pdf(two)
    def box(b):
        return NS(to_top_left_origin=lambda height:NS(l=b['x0'],t=b['y0'],r=b['x1'],b=b['y1']))
    model=NS(pages={1:NS(size=NS(width=700,height=400))},tables=[NS(
        prov=[NS(page_no=1,bbox=box(t['bbox']))],data=NS(num_rows=2,num_cols=2,table_cells=[NS(
            start_row_offset_idx=c['row_index'],start_col_offset_idx=c['column_index'],
            row_span=c['row_span'],col_span=c['column_span'],text=c['raw_text'],
            column_header=c['is_header'],bbox=box(c['bbox'])) for c in t['cells']]))])
    seen=[]
    def convert(stream,**kwargs):
        with fitz.open(stream=stream.stream.getvalue(),filetype='pdf') as extracted:
            assert len(extracted)==1
            seen.append(extracted[0].get_text())
        return NS(status=NS(value='success'),document=model)
    monkeypatch.setattr(table_docling,'_get_converter',lambda:NS(convert=convert))
    first=table_docling.read_page_result(combined.tobytes(),1)
    second=table_docling.read_page_result(combined.tobytes(),2)
    assert first['status']==second['status']=='complete'
    assert first['tables'][0]['rows'][-1][-1]=='0.00'
    assert second['tables'][0]['rows'][-1][-1]=='55.00'
    assert '55.00' not in seen[0] and '55.00' in seen[1]
    for index,result in [(1,first),(2,second)]:
        assert result['tables'][0]['page_index']==index
        assert all(c['page_index']==index for c in result['tables'][0]['cells'])


@pytest.mark.parametrize('clip', ['right', 'bottom'])
def test_clipped_table_cannot_prove_empty_terminal_column(clip):
    doc,t=header_fixture(extra=clip=='right')
    if clip=='right':t['bbox']['x1']=530
    else:doc[0].insert_text((550,180),'999.00',fontsize=8)
    result,u=refine_financial_boundaries(doc[0],[t])
    assert result[0]['col_count']==3 and u==1


@pytest.mark.parametrize('x,text', [(240,'-'),(240,'9'),(282,'-')])
def test_separate_source_sign_or_digit_cannot_be_dropped(x,text):
    doc,t,c=rule_fixture('123.45');before=deepcopy(c)
    doc[0].insert_text((x,90),text,fontsize=8)
    result,u=refine_financial_boundaries(doc[0],[t])
    assert result[0]['cells'][1]==before
    # Tokens inside the model bbox already invalidate exact source matching.
    assert result[0]['rows'][1][0]!='123.45'
    if x==240:assert u==1


@pytest.mark.parametrize('gain_x,cost_x,amount_x', [(300,540,575),(540,280,300)])
def test_summary_requires_native_column_ownership(gain_x,cost_x,amount_x):
    doc=fitz.open();p=doc.new_page(width=700,height=400);cells=[]
    add(p,cells,0,0,'Proceeds (Box 1d)',20,40,True)
    gain_col=1 if gain_x<cost_x else 2
    add(p,cells,0,gain_col,'Gain/Loss (-)',gain_x,40,True)
    add(p,cells,0,3-gain_col,'Cost basis (Box 1e)',cost_x,40,True)
    add(p,cells,1,0,'Box B Short-Term Realized Gain',20,80)
    a=add(p,cells,1,gain_col,'100.00',amount_x,80)
    p.insert_text((20,92),'Box B Short-Term Realized Loss',fontsize=8)
    p.insert_text((a['bbox']['x1']-fitz.get_text_length('55.00',fontsize=8),92),'55.00',fontsize=8)
    result,u=refine_financial_boundaries(p,[table(cells,2,3,110)])
    assert result[0]['row_count']==2 and u==1


def test_summary_source_owned_by_another_table_is_not_duplicated():
    doc,t=summary_fixture(loss_value='55.00')
    ws=[w for w in doc[0].get_text('words') if w[1]>80]
    other=[]
    for original,words in [(t['cells'][2],[w for w in ws if w[4]!='55.00']),
                           (t['cells'][3],[w for w in ws if w[4]=='55.00'])]:
        c=deepcopy(original)
        c.update(row_index=0,text=' '.join(w[4] for w in words),raw_text=' '.join(w[4] for w in words),
                 bbox=dict(x0=min(w[0] for w in words),y0=min(w[1] for w in words),
                           x1=max(w[2] for w in words),y1=max(w[3] for w in words)))
        other.append(c)
    second=table(other,1,2,110)
    for inputs in [[t,second],[second,t]]:
        result,_=refine_financial_boundaries(doc[0],inputs)
        assert sorted(x['row_count'] for x in result)==[1,2]


@pytest.mark.parametrize('reason',['hidden','rotated','word_limit'])
def test_unavailable_source_never_certifies_known_merged_columns(reason):
    doc,t=header_fixture()
    if reason=='hidden':doc[0].insert_text((20,200),'HIDDEN',render_mode=3)
    elif reason=='rotated':doc[0].insert_text((20,200),'SIDEWAYS',rotate=90)
    else:
        # A source wrapper supplies actual native words beyond the safety cap.
        class ExcessWords:
            rotation=0
            rotation_matrix=fitz.Matrix(1,1)
            def get_texttrace(self):return doc[0].get_texttrace()
            def get_text(self,kind):return doc[0].get_text(kind)*1000
        page=ExcessWords()
    result,u=refine_financial_boundaries(page if reason=='word_limit' else doc[0],[t])
    assert u==1 and result[0]['col_count']==3
    assert 'native_merged_financial_columns' in result[0]['warnings']


@pytest.mark.parametrize('size,offset',[(5,0),(8,1),(5,-1)])
@pytest.mark.parametrize('operation',['rule','new_summary','prior_summary'])
def test_small_or_shifted_sign_is_not_dropped(size,offset,operation):
    if operation=='rule':
        doc,t,c=rule_fixture('123.45')
        doc[0].insert_text((240,90+offset),'-',fontsize=size)
        result,u=refine_financial_boundaries(doc[0],[t])
        assert result[0]['rows'][1][0]!='123.45'
    else:
        doc,t=summary_fixture(loss_value='55.00')
        y=92 if operation=='new_summary' else 80
        value='55.00' if operation=='new_summary' else '100.00'
        x=t['cells'][-1]['bbox']['x1']-fitz.get_text_length(value,fontsize=8)-8
        doc[0].insert_text((x,y+offset),'-',fontsize=size)
        result,u=refine_financial_boundaries(doc[0],[t])
        assert result[0]['row_count']==2
    assert u==1
