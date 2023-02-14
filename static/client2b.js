var settingsopen = false ;

function main() {
  var app = new Vue({
	vuetify: new Vuetify(),   
	el : "#app",  
    data :
	  {
		status: '', 
		status2: '',
	    version: '',   
		text: '',
		ntext: '',
		mtext: '',
		nmtext: '',
		i:0,
		tot:25,
		beta:0.5, 
		steps:25,
		lr: 7.,
		seed:0,
		fgw:1,
		bgw:1,
		blend: 0,  
		imgpath: 'static/result.jpg',
		caption: '',
		gamma: 1.,   
		contrast: 1.,  
		num: 0,
		multi:false,
  	    hlist: ["512","640","768","960","1024"],
	    wlist: ["512","640","768","960","1024"],
	    modellist: ["-"],
	    schedlist: ["LMS", "EULER", "EULERv"],
	    h: 768,
	    w: 768,
	    model: "",
	    sched: "",
		inpaint: 0,
		  showmaskarea: false,  
		settingsdialog: false  		  
      },
	mounted() {
	  	 axios
		    .get("/data")
		    .then (function(response) {
		    	d = response.data
				if (d) {
					this.version = d.version
					this.status = d.status
	     	   		this.text = d.text
					this.i = d.i
					this.tot = d.n	
					this.ntext = d.ntext
					this.mtext = d.mtext
					this.nmtext = d.nmtext
					
					this.lr = d.lr
					this.beta = d.beta		
					this.steps = d.steps
					this.noise = d.noise
					this.gamma = d.gamma
					this.contrast = d.contrast
					this.bgw = d.bgw
					this.fgw = d. fgw
					this.blend = d.blend
					this.status = d.status
					this.status2 = d.status2
					this.h = d.h
					this.w = d.w
					this.model = d.model
					this.sched = d.sched	
					this.hlist = d.hlist
					this.wlist = d.wlist
					this.modellist = d.modellist
					this.schedlist = d.schedlist	
				    document.getElementById("version").innerHTML = d.version	
					document.getElementById("status").innerHTML = d.status	
					document.getElementById("status2").innerHTML = d.status2	
					document.querySelector("#text").value = d.text
					document.querySelector("#ntext").value = d.ntext		
					document.querySelector("#mtext").value = d.mtext
					document.querySelector("#nmtext").value = d.nmtext						
				}
	  		})

		 setTimeout(this.updater, 1000) ;
	  },
	methods: 
	      {
		  
	      /* send button handler */
	      do_send() {
	  		text = document.querySelector("#text").value
	  		ntext = document.querySelector("#ntext").value
			mtext = document.querySelector("#mtext").value  
			nmtext = document.querySelector("#nmtext").value
			axios
			  .post('/prompts', {
				  prompt: text,
				  nprompt: ntext,
				  mprompt: mtext,
				  nmprompt: nmtext
	    		})		
  	  		},
		
  	      /* go on button handler */
  	      do_next() {
  	  		text = document.querySelector("#text").value
  	  		ntext = document.querySelector("#ntext").value
			mtext = document.querySelector("#mtext").value  
			nmtext = document.querySelector("#nmtext").value
  			axios
  			  .post('/nexts', {
  				  prompt: text,
  				  nprompt: ntext,
				  mprompt: mtext,
				  nmprompt: nmtext
  	    		})		
    	      },
		
	        /* reset latents i.e. start diffusion from scratch */
			do_reset() {
  			axios
  			  .post('/reset', {
  	    		})					
			},
		
			/* handle old-new (beta) slider */
			do_beta() {
				val = document.querySelector("#beta").value
				axios
				   .post('/beta', {
				   	   beta: val
				   })		
			},

			do_lr() {
				val = document.querySelector("#lr").value
				axios
				   .post('/lr', {
				   	   lr: val
				   })		
			},
	
			do_steps() {
				val = document.querySelector("#steps").value
				axios
				   .post('/steps', {
				   	   steps: val
				   })		
			},
			
			do_init_img() {
				fn = document.querySelector("#initimgsel").files[0]
				var formData = new FormData();
				formData.append("file", fn);	
				//
				axios.post('/startimg',
  				formData, {
	           	 	headers: {
	                    'Content-Type': 'multipart/form-data'
	                }
	       	 	})
			},	

			do_mask_img() {
				fn = document.querySelector("#maskimgsel").files[0]
				var formData = new FormData();
				formData.append("file", fn);	
				//
				axios.post('/maskimg',
  				formData, {
	           	 	headers: {
	                    'Content-Type': 'multipart/form-data'
	                }
	       	 	})
			},	
			
	        /* reset mask */
			reset_mask() {
  			axios
  			  .post('/resetmask', {
  	    		})					
			},
		
			
			do_gamma()	{
					val = document.querySelector("#gamma").value
 					axios
 				   		.post('/gamma', {
 				   	   	 	gamma: val
 				    })		
				},
			
			do_noise()	{
					val = document.querySelector("#noise").value
 					axios
 				   		.post('/noise', {
 				   	   	 	noise: val
 				    })		
				},
			
			do_contrast()	{
					val = document.querySelector("#contrast").value
 					axios
 				   		.post('/contrast', {
 				   	   	 	contrast: val
 				    })		
				},
			
			do_bgw() {
				val = document.querySelector("#bgw").value
				axios
			   		.post('/bgw', {
			   	   	 	bgw: val
			    })		
			},

			do_fgw() {
				val = document.querySelector("#fgw").value
				axios
			   		.post('/fgw', {
			   	   	 	fgw: val
			    })		
			},
			
			do_blend() {
				val = document.querySelector("#blend").value
				axios
			   		.post('/blend', {
			   	   	 	blend: val
			    })		
			},
			
			do_inpaint() {
				val = app.inpaint
				val = val ? 1 : 0 ;
				axios
			   		.post('/inpaint', {
			   	   	 	inpaint: val
			    })		
			},
			
			open_settings() {
				settingsopen = true
			},
			
			cancel_settings() {
			    settingsopen = false	
			},
				
			do_settings() {
				//h = document.querySelector("#hsel").item-value
				//w = document.querySelector("#wsel").item-value
				//m = document.querySelector("#modelsel").item-value
				//s = document.querySelector("#schedsel").item-value
				settingsopen = false
				
				axios
		   		.post('/settings', {
		   	   	 	h: this.h,
					w: this.w,
					model: this.model,
					sched: this.sched
		    })	
			},
			
		    updater() {
			  	axios
			  	  .get("/data")
			        .then (function(response) {
						
					    function sic(o, p, n){
							changed = (n !== o[p])
					    	if (changed) {
								o[p] = n }
							return changed	
					    }
						
						function cleant(s) {
							s = s.replace("&lt;","<").replace("&gt;",">")
							return s							
						}
						
						vm = app
						
				      	d = response.data
				  		if (d) {
				     	   	sic(vm, 'text',  cleant(d.text))
				  			sic(vm, 'i',  d.i)
				  			sic(vm, 'tot',  d.n)
				  			sic(vm, 'ntext', cleant(d.ntext))
							sic(vm, 'mtext', cleant(d.mtext))
							sic(vm, 'nmtext', cleant(d.nmtext))
			
				  			if (settingsopen == false) 
				  			{
				  					c = sic(vm, 'h', d.h)
				  					c = sic(vm, 'w', d.w)	
				  					sic(vm, 'model', d.model)
				  					sic(vm, 'sched',d.sched)						
				  					sic(vm, 'modellist', d.modellist)
				  					sic(vm, 'schedlist', d.schedlist)
				  			}
	                    		
				  			c = sic(vm, 'status', d.status)
				  			if (c) {
				  				document.getElementById("status").innerHTML = d.status
				  			}
				  			c = sic(vm, 'status2', d.status2)
				  			if (c) {
				  				document.getElementById("status2").innerHTML = d.status2
				  			}
			
				  			if (vm.multi) {	
				  				sic(vm, 'lr', d.lr)
				  				sic(vm, 'beta', d.beta)		
				  				sic(vm, 'steps', d.steps)
				  				sic(vm, 'noise', d.noise)
				  				sic(vm, 'gamma', d.gamma)
				  				sic(vm, 'contrast', d.contrast)
				  				sic(vm, 'bgw', d.bgw)
				  				sic(vm, 'fgw', d.fgw)
				  				sic(vm, 'blend', d.blend)
				  			}

					  	    vm.caption = cleant(vm.text) + " (" + vm.i +  "/" + vm.tot + ")"				
					  	    var dt = new Date();
					  		vm.imgpath = "static/result.jpg?"+ dt.getTime();
	                      }			  					
				  	      setTimeout(vm.updater, 1000) ;
			      }	
	 		 )		  			  		
			}	  
	  }
  })
  
  function list2selectitems(l) {
  	
	  // convert list of simple variables (strings or numbers)
	  // into an object with if and value for v-select items
	
	  o = {}
	  for (i = 0; i < l.length; i++) {
	  	o.id = l[i]
		o.val = l[i]  
	  }
	 
	  return o
  }


   
}


