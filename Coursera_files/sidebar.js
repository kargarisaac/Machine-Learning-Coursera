define(["jquery","underscore","backbone","js/lib/q","js/lib/readme","pages/spark-survey/assessApi","pages/spark-survey/util/abUtils","pages/spark/app","i18n!pages/spark/views/template/nls/sidebar","pages/spark/views/template/sidebar.html","spark/app/signature/js/signature_track","bundles/assess/assessmentTypes/sparkSurveyQuestions/sparkSurveyQuestionsSessionModel","js/lib/coursera.ab","pages/help-center/data/HelpCenterLinks"],function($,_,Backbone,Q,Readme,assessApi,AbUtils,Coursera,_t,template,SigtrackFunctions,SparkSurveyQuestionsSessionModel,AB,HelpCenterLinks){var sidebar=Backbone.View.extend({name:"sidebar",className:"coursera-sidebar",attributes:{role:"menubar"},options:{},events:{"click .in-class-qqs-tab":"dismissPopup"},initialize:function(){var self=this;this.on("view:appended",function(){self.model.on("change:numQuestions",self.changeSideBarNumber,self)})},determineQuickQuestionTabPresence:function(){var self=this,endsWith=function(string,suffix){return-1!==string.indexOf(suffix,string.length-suffix.length)};if(!this.model)this.on("view:appended",this.getRemainingQuestions,this);else this.getRemainingQuestions()},getRemainingQuestions:function(){if(AbUtils.userAndSessionHaveSparkSurveyQuestions()){this.model.options=_(this.model.options||{}).extend({api:assessApi});var self=this;this.model.on("change:numQuestions",function(){var state=self.model.get("state");self.options.numQuestions=self.model.get("numQuestions"),self.options.hasRemainingQuestions=self.model.get("numQuestions")>0,self.trigger("qqsDetermined")},self),self.model.fetchCount()}else this.options.hasRemainingQuestions=!1,this.trigger("qqsDetermined")},dismissPopup:function(){$(".in-class-qqs-announcement [data-readme-close]")[0].click()},getQQLabel:function(){var experiment=AB.user.getExperiment("in_class_qqs_tab_label");switch(experiment.getChosenVariant()){case"default_wording":qqLabel="Quick Questions";break;case"polls_wording":qqLabel="Polls";break;case"feedback_wording":qqLabel="Feedback";break;case"give_feedback_wording":qqLabel="Give Feedback!";break;default:qqLabel="Quick Questions"}return qqLabel},renderAfterQuickQuestionsTabDetermined:function(renderQuickQuestionsTab,renderCallout,numQuestions){this.$el.html(template({_t:_t,course:Coursera.course,user:Coursera.user,navbar:Coursera.navbar,config:Coursera.config,url:encodeURIComponent(decodeURIComponent(window.location.href)),hasSparkSurveyQuestions:renderQuickQuestionsTab,hasSparkSurveyQuestionsCallout:renderCallout,hasSparkSurveyProgressNumber:AbUtils.showProgressNumber(),numQuestions:numQuestions,helpCenterLinks:HelpCenterLinks,qqLabel:this.getQQLabel()})),this.updateSidebar(),SigtrackFunctions.addLastChanceModalInteraction(this.$el),this.trigger("sidebarRendered")},render:function(){return Coursera.course.on("dateSet",this.determineQuickQuestionTabPresence,this),Coursera.course.on("datesNotFound",function(){this.renderAfterQuickQuestionsTabDetermined(!1,!1,this.options.numQuestions)},this),this.on("qqsDetermined",function(){this.renderAfterQuickQuestionsTabDetermined(this.options.hasRemainingQuestions,AbUtils.showTabCallout(),this.options.numQuestions)}),this.on("sidebarRendered",function(){if(AbUtils.showTabCallout())new Readme(".in-class-qqs-announcement [data-readme]")}),AbUtils.setSessionStartAndEndDates(),this},changeSideBarNumber:function(){var self=this;this.model.on("change:numQuestions",function(){self.$(".in-class-qqs-number-display").text(self.model.get("numQuestions"))})},updateSidebar:function(){function markLink(query){if(foundActiveLink)return;var matches=[];if($navbar.find(query).each(function(){matches.push($(this).parent()),foundActiveLink=!0}),1==matches.length)if(foundActiveLink=!0,-1==window.location.href.indexOf("/search?q"))matches[0].addClass("active"),matches[0].find("a").append('<span class="course-navbar-selected-marker">(selected)</span>')}this.$el.find(".course-navbar-container li").removeClass("active"),this.$el.find(".course-navbar-container li").remove(".course-navbar-selected-marker");var $navbar=this.$el.find(".course-navbar-container li"),foundActiveLink=!1;markLink('a[href="'+window.location.pathname+"/index"+window.location.search+'"]'),markLink('a[href="'+window.location.pathname+window.location.search+'"]'),markLink('a[href="'+window.location.href+'"]'),markLink('a[href="'+window.location.pathname+'"]'),markLink('a[href="'+window.location.href+'/index"]'),markLink('a[href="'+window.location.pathname+'/index"]'),markLink('a[href^="'+window.location.href.split("?")[0]+'"]')}});return sidebar});