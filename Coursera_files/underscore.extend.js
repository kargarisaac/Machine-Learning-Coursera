define(["underscore"],function(_){var courseraSimpleDeepClone=function(obj){if(_(obj).isArray())return _(obj).map(function(e){return courseraSimpleDeepClone(e)});else if(_(obj).isObject()&&!_(obj).isFunction()){var clone={};return _(obj).chain().keys().each(function(key){clone[key]=courseraSimpleDeepClone(obj[key])}).value(),clone}else return _(obj).clone()};return _.mixin({courseraPick:function(object,keys){var copy={};return _(keys).each(function(key){copy[key]=object[key]}),copy},courseraSimpleDeepClone:courseraSimpleDeepClone,courseraMapValues:function(input,mapper){return _(input).chain().map(function(v,k){return[k,mapper(v)]}).object().value()},courseraSum:function(input,extractor){var extractorFunction=extractor||_.identity;if(!_.isFunction(extractorFunction))extractorFunction=function(obj){return obj[extractor]};return _(input).chain().map(extractorFunction).reduce(function(a,b){return a+b},0).value()}}),_});