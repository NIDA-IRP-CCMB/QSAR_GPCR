select a.pref_name,
       a.organism,
       b.assay_id,
       b.assay_type,
       b.relationship_type,
       g.relationship_desc,
       b.confidence_score,
       b.curated_by, 
       b.description,
       c.activity_id,
       c.relation,
       c.value,
       c.units,
       c.type,
       c.standard_relation,
       c.standard_value,
       c.standard_units,
       c.standard_flag,
       c.standard_type,
       c.pchembl_value,
       c.activity_comment,
       c.data_validity_comment,
       c.potential_duplicate,
       c.text_value,
       c.standard_text_value,
       d.molregno,
       i.chembl_id,
       d.canonical_smiles,
       i.pref_name,
       h.parent_molregno,
       h.active_molregno,
       e.doc_id,
       e.pubmed_id,
       e.doi,
       e.journal,
       e.year,
       e.volume,
       e.first_page,
       f.src_short_name,        
       b.bao_format
from target_dictionary as a,
	assays as b,
	activities as c,
	compound_structures as d,
	docs as e,
	source as f,
	relationship_type as g, 
	molecule_hierarchy as h,
	molecule_dictionary as i
 where ( a.tid = 72 or
         a.tid = 11344 or
         a.tid = 11426 or
	 a.tid = 11427 or
	 a.tid = 14037 or 
	 a.tid = 104754 or
	 a.tid = 105041 or
	 a.tid = 105671 or
	 a.tid = 107904 or
	 a.tid = 104807 or
	 a.tid = 104847 or
	 a.tid = 104886 or
	 a.tid = 104906 or
	 a.tid = 104984 or
	 a.tid = 104994 or
	 a.tid = 105040 or
	 a.tid = 105104 or 
	 a.tid = 118327 or
         a.tid = 100027 ) and (
       c.activity_comment is null or
       c.activity_comment not in ( 'no data','Not available','Not Determined','Not done','not isolated','Not tested')) and
 	a.tid = b.tid and
 	b.assay_id = c.assay_id and
 	c.molregno = d.molregno and
 	b.doc_id = e.doc_id and
 	e.src_id = f.src_id and
 	b.relationship_type = g.relationship_type and
 	c.molregno = h.molregno and
 	c.molregno = i.molregno
