function scalar = load_timing_scalar(monkey_name)
%LOAD_TIMING_SCALAR Load the published mean wm for one monkey.

project_root = fileparts(mfilename('fullpath'));
scalar_file = fullfile(project_root, 'Data', 'Pre_Physiology', ...
    'Model_Results', 'Behavioral_Psychometrics', 'timing_scalars.mat');

if ~isfile(scalar_file)
    error('load_timing_scalar:MissingFile', ...
        'Timing scalar file not found: %s', scalar_file);
end

saved = load(scalar_file, 'timing_scalars');
if ~isfield(saved, 'timing_scalars') || ...
        ~isstruct(saved.timing_scalars) || ...
        ~isfield(saved.timing_scalars, monkey_name)
    error('load_timing_scalar:MissingMonkey', ...
        'No timing scalar found for %s in %s.', monkey_name, scalar_file);
end

scalar = saved.timing_scalars.(monkey_name);
if ~(isnumeric(scalar) && isscalar(scalar) && isfinite(scalar) && scalar > 0)
    error('load_timing_scalar:InvalidValue', ...
        'Invalid timing scalar for %s in %s.', monkey_name, scalar_file);
end
end
