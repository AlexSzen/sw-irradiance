;Get HMI vector magetic field data locally and make it the same
;FOV as AIA. Add header to original files.  Then rescale the data to 512x512. 
; Input: t0 (start time), t1 (end time)
; Output: HMI vector fits files in 512x512 size.
; WRITTEN: Meng Jin@LMSAL

pro get_hmi_vector,t0=t0,t1=t1

ssw_jsoc_time2data, t0, t1, index,fnames,$
                   ds='hmi.B_720s',$
                   /jsoc2,/silent,/local_files,/files_only

MM=n_elements(fnames)

for i=0,MM-1 do begin

   f_loc=STRSPLIT(fnames[i],'/',/EXTRACT)
   file_inclination=fnames[i]
   file_azimuth='/'+f_loc[0]+'/'+f_loc[1]+'/'+f_loc[2]+'/'+'azimuth.fits'
   file_field='/'+f_loc[0]+'/'+f_loc[1]+'/'+f_loc[2]+'/'+'field.fits'

   read_sdo,file_inclination,oindex,odata,/uncomp_delete
   odata[where(odata lt 0)]=0.
   aia_prep,index[i],odata,nindex,ndata,/interp
   mwritefits,nindex,ndata,outfile='temp.fits'
   read_sdo,'temp.fits',findex,fdata,outsize=[512,512],/uncomp_delete
   ftemp=strsplit(findex.T_REC,'_',escape='.:',/extract)
   mwritefits,findex,fdata,outfile='hmi.M_720s.'+ftemp[0]+'_'+ftemp[1]+'_inclination.fits'

   read_sdo,file_azimuth,oindex,odata,/uncomp_delete
   odata[where(odata lt 0)]=0.
   aia_prep,index[i],odata,nindex,ndata,/interp
   mwritefits,nindex,ndata,outfile='temp.fits'
   read_sdo,'temp.fits',findex,fdata,outsize=[512,512],/uncomp_delete
   ftemp=strsplit(findex.T_REC,'_',escape='.:',/extract)
   mwritefits,findex,fdata,outfile='hmi.M_720s.'+ftemp[0]+'_'+ftemp[1]+'_azimuth.fits'

   read_sdo,file_field,oindex,odata,/uncomp_delete
   odata[where(odata lt 0)]=0.
   aia_prep,index[i],odata,nindex,ndata,/interp
   mwritefits,nindex,ndata,outfile='temp.fits'
   read_sdo,'temp.fits',findex,fdata,outsize=[512,512],/uncomp_delete
   ftemp=strsplit(findex.T_REC,'_',escape='.:',/extract)
   mwritefits,findex,fdata,outfile='hmi.M_720s.'+ftemp[0]+'_'+ftemp[1]+'_field.fits'

endfor

end
