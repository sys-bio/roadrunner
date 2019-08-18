%module telplugins

%{
	#include "telAPIHandleManager.h"
	#include "telplugins_c_api.h"
	#include "telplugins_properties_api.h"
	#include "telplugins_telluriumdata_api.h"
	#include "telplugins_matrix_api.h"
	#include "telplugins_cpp_support.h"
	#include "telplugins_utilities.h"
	#include "telplugins_logging_api.h"
	#include "telAPIHandleManager.h"
	#include "telplugins_c_api.h"
%}
%include "windows.i";

%include "telplugins_exporter.h";
%include "telplugins_settings.h";
%include "tel_macros.h"
%include "telplugins_types.h";

%include "telplugins_c_api.h";